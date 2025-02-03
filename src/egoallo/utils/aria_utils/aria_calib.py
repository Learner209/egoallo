# project_aria.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
from projectaria_tools.core import calibration
from projectaria_tools.core.image import InterpolationMethod
from egoallo.utils.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)

# # Calibration examples
# Goal:
# - Obtain camera extrinsics and intrinsics
# - Learn to project a 3D point to camera frame
#
# Key learnings
# - Get calibration for different sensors using sensor labels
# - Learn how to use extrinsics/intrinsics to project a 3D points to a given camera
# - Reference frame convention


class CalibrationUtilities:
    def __init__(self, provider):
        self.provider = provider
        self.device_calib = provider.get_calibration()

    def get_camera_calibration(self, camera_label):
        return self.device_calib.get_camera_calib(camera_label)

    def get_base_frame_sensor_lable(self):
        return self.device_calib.get_origin_label()

    def project_points_w_device_base_to_cam_frame(self):
        # ### Project a 3D point in the aria device base frame to camera frame specified by the camera_name/cam_label
        #
        # In this section we will learn how to retrieve calibration data and how to use it.
        # Aria calibration is defined by two objects: one defining the intrinsics (`rgb_calib.project` and `rgb_calib.unproject`) and one defining the extrinsics as a SE3 pose (`device_calib.get_transform_device_sensor(sensor_label`).
        #
        # Intrinsics can be used to project a 3d point to the image plane or un-project a 2d point as a bearing vector. Extrinsics are used to set the camera in world coordinates at a given rotation and position in space.
        #
        # ### Reference frame convention
        #
        # > `transform_sensor1_sensor3` = `transform_sensor1_sensor2` * `transform_sensor2_sensor3` \
        # > `point_in_sensor`: 3D point measured from sensor's reference frame \
        # > `point_in_sensor` = `transform_sensor1_sensor` * `point_in_sensor`
        #
        # Device Frame: `device_calib.get_origin_label() = camera-slam-left`\
        # Sensor extrinsics: `device_calib.get_transform_device_sensor(sensor_label)`

        camera_name = "camera-rgb"
        transform_device_camera = self.device_calib.get_transform_device_sensor(
            camera_name
        ).to_matrix()
        transform_camera_device = np.linalg.inv(transform_device_camera)
        print(f"Device calibration origin label {self.device_calib.get_origin_label()}")
        print(f"{camera_name} has extrinsics of \n {transform_device_camera}")

        rgb_calib = self.device_calib.get_camera_calib("camera-rgb")
        if rgb_calib is not None:
            # project a 3D point in device frame [camera-slam-left] to rgb camera
            point_in_device = np.array([0, 0, 10])
            point_in_camera = (
                np.matmul(
                    transform_camera_device[0:3, 0:3], point_in_device.transpose()
                )
                + transform_camera_device[0:3, 3]
            )

            maybe_pixel = rgb_calib.project(point_in_camera)
            if maybe_pixel is not None:
                print(
                    f"Get pixel {maybe_pixel} within image of size {rgb_calib.get_image_size()}"
                )

    def get_calib_for_sensor_w_device_base(self):
        # ### Get calibration data for other sensors
        # Aria is a multimodal capture device, each sensors calibration can be retrieved using the same interface. Only EyeTracking (`get_aria_et_camera_calib()`) and Audio calibration (`get_aria_microphone_calib()`) is a bit different since we have multiple sensors that share the same stream_id.

        et_calib = self.device_calib.get_aria_et_camera_calib()
        if et_calib is not None:
            print(
                f"Camera {et_calib[0].get_label()} has image size {et_calib[0].get_image_size()}"
            )
            print(
                f"Camera {et_calib[1].get_label()} has image size {et_calib[1].get_image_size()}"
            )

        imu_calib = self.device_calib.get_imu_calib("imu-left")
        if imu_calib is not None:
            print(
                f"{imu_calib.get_label()} has extrinsics transform_Device_Imu:\n {imu_calib.get_transform_device_imu().to_matrix3x4()}"
            )

    def undistort_aria_image(self):
        # ### Undistort an image
        # You can remove distortions in an image in three steps.
        #
        # First, use the provider to access the image and the camera calibration of the stream.
        # Then create a "linear" spherical camera model with `get_spherical_camera_calibration`.
        # The function allows you to specify the image size as well as focal length of the model,
        # assuming principal point is at the image center. Finally, apply `distort_by_calibration` function to distort the image.

        # input: retrieve image as a numpy array
        cam_name = "camera-rgb"
        sensor_stream_id = self.provider.get_stream_id_from_label(cam_name)
        image_data = self.provider.get_image_data_by_index(sensor_stream_id, 0)
        image_array = image_data[0].to_numpy_array()
        # input: retrieve image distortion
        self.device_calib = self.provider.get_self.device_calibration()
        src_calib = self.device_calib.get_camera_calib(cam_name)

        # create output calibration: a linear model of image size 512x512 and focal length 150
        # Invisible pixels are shown as black.
        dst_calib = calibration.get_linear_camera_calibration(512, 512, 150, cam_name)

        # distort image
        rectified_array = calibration.distort_by_calibration(
            image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR
        )

        # visualize input and results
        plt.figure()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Image undistortion (focal length = {dst_calib.get_focal_lengths()})"
        )

        axes[0].imshow(image_array, cmap="gray", vmin=0, vmax=255)
        axes[0].title.set_text(f"sensor image ({cam_name})")
        axes[0].tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        axes[1].imshow(rectified_array, cmap="gray", vmin=0, vmax=255)
        axes[1].title.set_text(f"undistorted image ({cam_name})")
        axes[1].tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.show()

        # Note the rectified image shows a circular area of visible pixels. If you want the entire rectified image to be covered by pixels, you can increase the magnification.

    @staticmethod
    def get_exo_cam_masks(
        take, exo_traj_df, portrait_view=False, dimension=(3840, 2160)
    ):
        """
        Generate masks for exo cameras in the undistorted aria view.

        Parameters
        ----------
        take : dict
            Take dictionary.
        exo_traj_df : pd.DataFrame
            Dataframe containing the exo camera calibration data.
        portrait_view : bool, default=True
            Whether the view is portrait or landscape.
        dimesion : tuple, default=(3840, 2160)
            Dimension of the image. (W, H)

        Returns
        -------
        exo_cam_masks : dict
            Dictionary containing the masks for the exo cameras, each value is np.array of shape (H, W) ranging from [0,1].
        exo_cam_names : list

        Notes
        -----
        - The `valid` exo cams are the ones that are present in the `exo_traj_df`.

        """
        exo_cam_masks = {}
        if exo_traj_df is None:
            return exo_cam_masks, []
        valid_exo_cam_names = exo_traj_df.cam_uid.unique()
        if len(valid_exo_cam_names) == 0:
            return exo_cam_masks, []
        for ego_exo_cam_name in valid_exo_cam_names:
            calib_df = exo_traj_df[exo_traj_df.cam_uid == ego_exo_cam_name]
            if len(calib_df) == 0:
                continue
            calib_df = calib_df.iloc[0].to_dict()
            D, I = CalibrationUtilities.get_distortion_and_intrinsics(calib_df)
            # Generate mask in undistorted aria view
            mask = np.full(dimension[::-1], 255, dtype=np.uint8)
            undistorted_mask, new_K_latest = CalibrationUtilities.undistort_exocam(
                mask, I, D, dimension
            )
            undistorted_mask = (
                cv2.rotate(undistorted_mask, cv2.ROTATE_90_CLOCKWISE)
                if portrait_view
                else undistorted_mask
            )
            undistorted_mask = undistorted_mask / 255
            exo_cam_masks[ego_exo_cam_name] = undistorted_mask

        if len(exo_cam_masks) == 0:
            logger.warning(
                "No exo camera masks generated. Probably because there is a mismatch between take-file specified exo-cam_names and the goprocalibs.csv"
            )

        return exo_cam_masks, valid_exo_cam_names

    @staticmethod
    def undistort_exocam(image, intrinsics, distortion_coeffs, dimension=(3840, 2160)):
        """
        Undistorts the input image using camera intrinsic and distortion parameters.

        Parameters
        ----------
        image (numpy.ndarray of (*H*,*W*)) : The input image to be undistorted.
        intrinsics (numpy.ndarray) : The camera intrinsic parameters.
        distortion_coeffs (numpy.ndarray) : The distortion coefficients.
        dimension (tuple, optional) : The (W, H) of the undistorted image. Defaults to (3840, 2160).

        Returns
        --------
        tuple : A tuple containing the undistorted image and the new camera matrix.

        Raises
        -------
        AssertionError : If the aspect ratio of the input image does not match the aspect ratio used in calibration.
        """

        DIM = dimension
        dim2 = None
        dim3 = None
        balance = 0.8

        dim1 = image.shape[:2][
            ::-1
        ]  # dim1 is the dimension of input image to un-distort

        # Change the calibration dim dynamically (bouldering cam01 and cam04 are verticall for examples)
        if DIM[0] != dim1[0]:
            DIM = (DIM[1], DIM[0])

        assert dim1[0] / dim1[1] == DIM[0] / DIM[1], (
            "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        )
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = (
            intrinsics * dim1[0] / DIM[0]
        )  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            scaled_K, distortion_coeffs, dim2, np.eye(3), balance=balance
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv2.CV_16SC2
        )
        undistorted_image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return undistorted_image, new_K

    @staticmethod
    def get_distortion_and_intrinsics(_raw_camera):
        intrinsics = np.array(
            [
                [_raw_camera["intrinsics_0"], 0, _raw_camera["intrinsics_2"]],
                [0, _raw_camera["intrinsics_1"], _raw_camera["intrinsics_3"]],
                [0, 0, 1],
            ]
        )
        distortion_coeffs = np.array(
            [
                _raw_camera["intrinsics_4"],
                _raw_camera["intrinsics_5"],
                _raw_camera["intrinsics_6"],
                _raw_camera["intrinsics_7"],
            ]
        )
        return distortion_coeffs, intrinsics
