from egoallo.utils.setup_logger import setup_logger
from yacs.config import CfgNode as CN

logger = setup_logger(output=None, name=__name__)


def define_Dataset(dataset_opt: CN, opt: CN):
    dataset_type = dataset_opt["dataset_type"].lower()

    if dataset_type in ["egoexo"]:
        from data.dataset_egoexo import Dataset_EgoExo as D
    elif dataset_type in ["egoexo_inference"]:
        from data.dataset_egoexo import Dataset_EgoExo_inference as D
    elif dataset_type in ["filtered_egoexo"]:
        from data.filter_dataset_egoexo import Filtered_Dataset_EgoExo as D
    elif dataset_type in ["filtered_egoexo_inference"]:
        from data.filter_dataset_egoexo import Filtered_Dataset_EgoExo_inference as D
    elif dataset_type in ["egoexo_diffusion"]:
        from egoallo.data.datasets.egoexo_diffusion_dataset import (
            EgoExoDiffusionDataset as D,
        )
    # elif dataset_type in ['egoexo_diffusion_inference']:
    #     from data.egoexo_diffusion_dataset import EgoExoDiffusionDataset as D

    else:
        raise NotImplementedError("Dataset [{:s}] is not found.".format(dataset_type))

    if "diffusion" in dataset_type:
        dataset = D(opt, dataset_opt)
    else:
        dataset = D(dataset_opt)
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__,
            dataset_opt["phase"],
        ),
    )
    return dataset
