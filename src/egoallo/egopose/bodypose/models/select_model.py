import functools

import torch
from egoallo.utils.setup_logger import setup_logger
from torch.nn import init
# from human_body_prior.body_model.body_model import BodyModel

logger = setup_logger(output=None, name=__name__)


def define_Model(opt):
    model = opt["model"]  # one input: L

    if model == "egoexo":  # two inputs: L, C
        from models.model_egoexo import ModelEgoExo4D as M

    else:
        raise NotImplementedError("Model [{:s}] is not defined.".format(model))

    m = M(opt)

    logger.info("Training model [{:s}] is created.".format(m.__class__.__name__))
    return m


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------


def define_G(opt):
    opt_net = opt["netG"]
    net_type = opt_net["net_type"]
    device = torch.device("cuda" if opt["gpu_ids"] else "cpu")

    if net_type == "EgoExo4D":
        from models.network import EgoExo4D as net

        netG = net(
            input_dim=opt_net["input_dim"],
            output_dim=opt_net["output_dim"],
            num_layer=opt_net["num_layer"],
            embed_dim=opt_net["embed_dim"],
            nhead=opt_net["nhead"],
            device=device,
            opt=opt,
        )

    else:
        raise NotImplementedError("netG [{:s}] is not found.".format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt["is_train"]:
        init_weights(
            netG,
            init_type=opt_net["init_type"],
            init_bn_type=opt_net["init_bn_type"],
            gain=opt_net["init_gain"],
        )

    return netG


def init_weights(net, init_type="xavier_uniform", init_bn_type="uniform", gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type="xavier_uniform", init_bn_type="uniform", gain=1):
        classname = m.__class__.__name__

        if classname.find("Conv") != -1 or classname.find("Linear") != -1:
            if init_type == "normal":
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == "uniform":
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == "xavier_normal":
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == "xavier_uniform":
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == "kaiming_normal":
                init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == "kaiming_uniform":
                init.kaiming_uniform_(
                    m.weight.data,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                m.weight.data.mul_(gain)

            elif init_type == "orthogonal":
                # init.orthogonal_(m.weight.data, gain=gain)
                pass
            else:
                raise NotImplementedError(
                    "Initialization method [{:s}] is not implemented".format(init_type),
                )

        #            if m.bias is not None:
        #                m.bias.data.zero_()

        elif classname.find("BatchNorm2d") != -1:
            if init_bn_type == "uniform":  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == "constant":
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    "Initialization method [{:s}] is not implemented".format(
                        init_bn_type,
                    ),
                )

    if init_type not in ["default", "none"]:
        logger.info(
            "Initialization method [{:s} + {:s}], gain is [{:.2f}]".format(
                init_type,
                init_bn_type,
                gain,
            ),
        )
        fn = functools.partial(
            init_fn,
            init_type=init_type,
            init_bn_type=init_bn_type,
            gain=gain,
        )
        net.apply(fn)
    else:
        logger.info(
            "Pass this initialization! Initialization was done during network defination!",
        )
