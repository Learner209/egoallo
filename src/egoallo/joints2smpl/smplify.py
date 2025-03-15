import os
import pickle

import smplx
import torch
from egoallo.joints2smpl import joints2smpl_config
from egoallo.joints2smpl.customloss import body_fitting_loss_3d
from egoallo.joints2smpl.customloss import camera_fitting_loss_3d
from egoallo.joints2smpl.prior import MaxMixturePrior
from jaxtyping import Float
from torch import Tensor


@torch.no_grad()
# @jaxtyped(typechecker=typeguard.typechecked)
def guess_init_3d(
    j3d_est: Float[Tensor, "batch_size 45 3"],
    j3d_gt: Float[Tensor, "batch_size 22 3"],
    joints_category="orig",
) -> Float[Tensor, "batch_size 3"]:
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four
    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [joints2smpl_config.JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "orig":
        joints_ind_category = [
            joints2smpl_config.JOINT_MAP[joint] for joint in gt_joints
        ]
    elif joints_category == "AMASS":
        joints_ind_category = [
            joints2smpl_config.AMASS_JOINT_MAP[joint] for joint in gt_joints
        ]
    else:
        print("NO SUCH JOINTS CATEGORY!")

    sum_init_t = (j3d_gt[:, joints_ind_category] - j3d_est[:, gt_joints_ind]).sum(dim=1)
    init_t = sum_init_t / 4.0
    return init_t


# SMPLIfy 3D
class SMPLify3D:
    """Implementation of SMPLify, use 3D joints."""

    def __init__(
        self,
        smplxmodel: smplx.SMPL,
        step_size=1e-2,
        batch_size=1,
        num_iters=100,
        use_collision=False,
        use_lbfgs=True,
        joints_category="orig",
        device=torch.device("cuda:0"),
    ):
        # Store options
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # --- choose optimizer
        self.use_lbfgs = use_lbfgs
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(
            prior_folder=joints2smpl_config.GMM_MODEL_DIR,
            num_gaussians=8,
            dtype=torch.float32,
        ).to(device)
        # collision part
        self.use_collision = use_collision
        if self.use_collision:
            self.part_segm_fn = joints2smpl_config.Part_Seg_DIR

        # reLoad SMPL-X model
        self.smpl = smplxmodel

        self.model_faces = smplxmodel.faces_tensor.view(-1)

        # select joint joint_category
        self.joints_category = joints_category

        if joints_category == "orig":
            self.smpl_index = joints2smpl_config.full_smpl_idx
            self.corr_index = joints2smpl_config.full_smpl_idx
        elif joints_category == "AMASS":
            self.smpl_index = joints2smpl_config.amass_smpl_idx
            self.corr_index = joints2smpl_config.amass_idx
        else:
            self.smpl_index = None
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")

    # ---- get the man function here ------
    # @jaxtyped(typechecker=typeguard.typechecked)
    def __call__(
        self,
        init_pose: Float[Tensor, "batch_size 72"],
        init_betas: Float[Tensor, "batch_size 10"],
        init_cam_t: Float[Tensor, "batch_size 3"],
        j3d: Float[Tensor, "batch_size 22 3"],
        conf_3d: Float[Tensor, "22"],
        seq_ind: int = 0,
    ):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints
            conf_3d: confidence for 3d joints
            seq_ind: index of the sequence
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """

        # # # add the mesh inter-section to avoid
        search_tree = None
        pen_distance = None
        filter_faces = None

        if self.use_collision:
            # FIXME: solve for undefined imports of these modules.
            from egoallo.joints2smpl.mesh_intersection.bvh_search_tree import BVH
            import egoallo.joints2smpl.mesh_intersection.loss as collisions_loss
            from egoallo.joints2smpl.mesh_intersection.filter_faces import FilterFaces

            search_tree = BVH(max_collisions=8)

            pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
                sigma=0.5,
                point2plane=False,
                vectorized=True,
                penalize_outside=True,
            )

            if self.part_segm_fn:
                # Read the part segmentation
                part_segm_fn = os.path.expandvars(self.part_segm_fn)
                with open(part_segm_fn, "rb") as faces_parents_file:
                    face_segm_data = pickle.load(faces_parents_file, encoding="latin1")
                faces_segm = face_segm_data["segm"]
                faces_parents = face_segm_data["parents"]
                # Create the module used to filter invalid collision pairs
                filter_faces = FilterFaces(
                    faces_segm=faces_segm,
                    faces_parents=faces_parents,
                    ign_part_pairs=None,
                ).to(device=self.device)

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
        )
        model_joints = smpl_output.joints

        init_cam_t = guess_init_3d(
            model_joints,
            j3d,
            self.joints_category,
        ).detach()  # get the triangle distance averaged over four joints between initiial estimates and current joints cooridnate.
        camera_translation = init_cam_t.clone()

        preserve_pose = init_pose[:, 3:].detach().clone()
        # -------------Step 1: Optimize camera translation and body orientation--------
        # 1.Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(
                camera_opt_params,
                max_iter=self.num_iters,
                lr=self.step_size,
                line_search_fn="strong_wolfe",
            )
            for i in range(10):

                def closure():
                    camera_optimizer.zero_grad()
                    smpl_output = self.smpl(
                        global_orient=global_orient,
                        body_pose=body_pose,
                        betas=betas,
                    )
                    model_joints = smpl_output.joints

                    loss = camera_fitting_loss_3d(
                        model_joints,
                        camera_translation,
                        init_cam_t,
                        j3d,
                        self.joints_category,
                    )
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(
                camera_opt_params,
                lr=self.step_size,
                betas=(0.9, 0.999),
            )

            for i in range(20):
                smpl_output = self.smpl(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas,
                )
                model_joints = smpl_output.joints

                loss = camera_fitting_loss_3d(
                    model_joints[:, self.smpl_index],
                    camera_translation,
                    init_cam_t,
                    j3d[:, self.corr_index],
                    self.joints_category,
                )
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # 2.Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        # --- if we use the sequence, fix the shape
        if seq_ind == 0:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera_translation]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation]

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(
                body_opt_params,
                max_iter=self.num_iters,
                lr=self.step_size,
                line_search_fn="strong_wolfe",
            )
            for i in range(self.num_iters):

                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smpl(
                        global_orient=global_orient,
                        body_pose=body_pose,
                        betas=betas,
                    )
                    model_joints = smpl_output.joints
                    model_vertices = smpl_output.vertices

                    loss = body_fitting_loss_3d(
                        body_pose,
                        preserve_pose,
                        betas,
                        model_joints[:, self.smpl_index],
                        camera_translation,
                        j3d[:, self.corr_index],
                        self.pose_prior,
                        joints3d_conf=conf_3d,
                        joint_loss_weight=600.0,
                        pose_preserve_weight=5.0,
                        use_collision=self.use_collision,
                        model_vertices=model_vertices,
                        model_faces=self.model_faces,
                        search_tree=search_tree,
                        pen_distance=pen_distance,
                        filter_faces=filter_faces,
                    )
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(
                body_opt_params,
                lr=self.step_size,
                betas=(0.9, 0.999),
            )

            for i in range(self.num_iters):
                smpl_output = self.smpl(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas,
                )
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    betas,
                    model_joints[:, self.smpl_index],
                    camera_translation,
                    j3d[:, self.corr_index],
                    self.pose_prior,
                    joints3d_conf=conf_3d,
                    joint_loss_weight=600.0,
                    use_collision=self.use_collision,
                    model_vertices=model_vertices,
                    model_faces=self.model_faces,
                    search_tree=search_tree,
                    pen_distance=pen_distance,
                    filter_faces=filter_faces,
                )
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=True,
            )
            model_joints = smpl_output.joints
            model_vertices = smpl_output.vertices

            final_loss = body_fitting_loss_3d(
                body_pose,
                preserve_pose,
                betas,
                model_joints[:, self.smpl_index],
                camera_translation,
                j3d[:, self.corr_index],
                self.pose_prior,
                joints3d_conf=conf_3d,
                joint_loss_weight=600.0,
                use_collision=self.use_collision,
                model_vertices=model_vertices,
                model_faces=self.model_faces,
                search_tree=search_tree,
                pen_distance=pen_distance,
                filter_faces=filter_faces,
            )

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, final_loss
