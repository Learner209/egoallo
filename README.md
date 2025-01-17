# egoallo new

## Overview

**TLDR;** We use egocentric SLAM poses and images to estimate 3D human body pose, height, and hands.

This repository is structured as follows (partially illustrated below):

```
.
├── download_checkpoint_and_data.sh
│                            			- Download model checkpoint and sample data.
├── 1_train_motion_prior.py
│                            			- Training script for motion diffusion model.
├── src/egoallo/
│   ├── data/                			- Dataset utilities.
| 	|	|── amass/           			- OOM module in charge of preprocessing of AMASS dataset.
| 	|	|── hps/             			- OOM module in charge of preprocessing of HPS dataset.
| 	|	|── rich/            			- OOM module in charge of preprocessing of RICH dataset.
| 	|	|── collators/       			- the batch collator family functions of the dataloader.
| 	|	|── datasets/        			- Dataloader.
| 	| 	|	|── egoexo_dataset.py       - dataloading of EgoExoDataset (large **DEPRECATED** in favor of [](./src/egoallo/egopose/bodypose/data/dataset_egoexo.py)).
| 	| 	|	|── amass_dataset.py        - dataloading of AMASS dataset.
| 	|	|── rich/            			- the preprocessing of RICH dataset.
| 	├── egoexo/              			- The manipulation utilities of EgoEXoDataset dataset, mostly the EgoExoUtils module.
| 	├── egopose/             			- Egopose (EgoExoDataset) related utilities of EgoExoDataset. large borrowed from [egopose repo](https://github.com/EGO4D/ego-exo4d-egopose)
| 	|	|── bodypose/       			- Bodypose dataloading of EgoExoDataset.
| 	|	|── handpose/       			- Handpose dataloading of EgoExoDataset.
| 	| 	|	|── main_ego_pose.py        - The main script to extract valid egopose annotations from EgoExoDataset. Large Deprecated since this is used for [egoexo_dataset.py](./src/egoallo/data/datasets/egoexo_dataset.py), which is large deprecated as stated above.
| 	├── evaluation/          			- OOP implementation of evaluation metric utilities.
| 	├── joints2smpl/         			- Package used to convert sequences of joints positions to parameterized SMPL pose parameters. (large borrowed from [joints2smpl repo](https://github.com/wangsen1312/joints2smpl)). This package is no longer used, it is previously used in the joints' conversion to smpl parameterized poses.
| 	├── registry/            			- package to register a module, serve as a hook to load the module.
| 	├── scripts/             			- scripts of all related operations (e.g., training, inference, visualization, export dataset to hdf5 format etc.)
| 	|	|── aria_inference.py          - AriaInference class to some functionalities of the EgoExoDataset dataset.
| 	|	|── export_hdf5.py            - Load the preprocessed npz files of specified datasets paths and exprot them into a hdf5 format file.
| 	|	|── hamer_on_vrs.py           - run HaMeR hand detection on VRS data.
| 	|	|── run_hamer_on_vrs.py       - run HaMeR hand detection on VRS data.
| 	|	|── preprocess_amass.py       - the preprocessing **main script** of AMASS dataset.
| 	|	|── preprocess_hps.py         - the preprocessing **main script** of HPS dataset.
| 	|	|── preprocess_rich.py        - the preprocessing **main script** of RICH dataset.
| 	|	|── test.py                   - the inference **main script** of the model.
| 	|	|── visualize_inference.py    - the main script of visualization of the model outputs, serve as callee for test.py, called as a separate process, to avoid GPU OOM. 
| 	|	|── visualize_outputs.py      - not used.
│   ├── transforms/                  - SO(3) / SE(3) transformation helpers.
│   ├── utils/                      - utility package.
│   ├── viz/                        - some of the rendering pipeline of the opengl rendering.
│   └── *.py                        - other implementations.
│
├── third_party/
│   ├── cloudrender                 - opengl rendering pipelines.
│   ├── hamer                       - HaMeR hand detection.
└── pyproject.toml                  - Python dependencies/package metadata.
└── train_motion_prior.py           - the **main script** of training the motion diffusion model.

## Getting started

EgoAllo requires Python 3.12 or newer.

1. **Clone the repository.**
   ```bash
   git clone https://github.com/Learner209/egoallo.git
   ```
2. **Install general dependencies.**
   ```bash
   cd egoallo
   pip install -e .
   ```
3. **Download+unzip model checkpoint and sample data.**

   ```bash
   bash download_checkpoint_and_data.sh
   ```

   You can also download the zip files manually: here are links to the [checkpoint](https://drive.google.com/file/d/14bDkWixFgo3U6dgyrCRmLoXSsXkrDA2w/view?usp=drive_link) and [example trajectories](https://drive.google.com/file/d/14zQ95NYxL4XIT7KIlFgAYTPCRITWxQqu/view?usp=drive_link).

4. **Download the SMPL-H model file.**

   You can find the "Extended SMPL+H model" from the [MANO project webpage](https://mano.is.tue.mpg.de/).
   Our scripts assumes an npz file located at `./data/smplh/neutral/model.npz`, but this can be overridden at the command-line (`--smplh-npz-path {your path}`).


## Running training

1. **Training the motion diffusion model.**

   To train the motion diffusion model, you can run:

   ```bash
   python train_motion_prior.py
   ```

   You can run `python train_motion_prior.py --help` to see the full list of options.

   The amass-rich-hps combined training data is uploaded to nas at '/public/tmp/egoallo/amass_rich_hps/processed_amass_rich_hps_correct.hdf5' and '/public/tmp/egoallo/amass_rich_hps/processed_amass_rich_hps_correct.txt' 
   The amass-only training data is uploaded to nas at '/public/tmp/egoallo/amass/processed_amass_correct.hdf5' and '/public/tmp/egoallo/amass/processed_amass_correct.txt'

   There are a lot of options for training, a full working example of running this script would be:

   ```bash
   python train_motion_prior.py --config.batch-size 64 --config.experiment-name <your-experiment-name> --config.learning-rate 1e-4 --config.dataset-hdf5-path <your-hdf5-path> --config.dataset-files-path <your-txt-path> --config.mask_ratio 0.9 --config.splits train val --config.joint_cond_mode "absrel" --config.use_fourier_in_masked_joints --config.random_sample_mask_ratio --config.data_collate_fn "TensorOnlyDataclassBatchCollator" --config.subseq_len 128
   ```

   Or, if you choose to use accelerate module, 
   ```bash
   accelerate launch train_motion_prior.py --config.batch-size 64 --config.experiment-name <your-experiment-name> --config.learning-rate 1e-4 --config.dataset-hdf5-path <your-hdf5-path> --config.dataset-files-path <your-txt-path> --config.mask_ratio 0.9 --config.splits train val --config.joint_cond_mode "absrel" --config.use_fourier_in_masked_joints --config.random_sample_mask_ratio --config.data_collate_fn "TensorOnlyDataclassBatchCollator" --config.subseq_len 128
   ```


## Running inference

1. **Installing inference dependencies.**

   Our guidance optimization uses a Levenberg-Marquardt optimizer that's implemented in JAX. If you want to run this on an NVIDIA GPU, you'll need to install JAX with CUDA support:

   ```bash
   # Also see: https://jax.readthedocs.io/en/latest/installation.html
   pip install -U "jax[cuda12]"
   ```

   You'll also need [jaxls](https://github.com/brentyi/jaxls):

   ```bash
   pip install git+https://github.com/brentyi/jaxls.git
   ```

2. **Running inference on example data.**

	The inference main script for **Amass** is located at './src/egoallo/scripts/test.py'. A full working example of using it would be:
	```bash
	 python ./src/egoallo/scripts/test.py --inference-config.dataset-slice-strategy full_sequence --inference-config.splits test --inference-config.checkpoint-dir <your-ckpt-path> --inference-config.dataset-type AdaptiveAmassHdf5Dataset --inference-config.visualize-traj
	```

	The inference main script for **EgoExo** is located at './src/egoallo/scripts/test.py'. A full working example of using it would be:
	```bash
	python ./src/egoallo/scripts/test.py --inference-config.dataset-slice-strategy full_sequence --inference-config.splits test --inference-config.checkpoint-dir <your-ckpt-path> --inference-config.dataset-type EgoExoDataset --inference-config.visualize-traj --inference-config.output-dir ./exp/test-egoexo-train
	```

3. **Running inference with post-processing.**

   To run inference with post-processing, you can use the `--inference-config.post-process` flag:

   ```bash
   python ./src/egoallo/scripts/test.py --inference-config.dataset-slice-strategy full_sequence --inference-config.splits test --inference-config.checkpoint-dir <your-ckpt-path> --inference-config.dataset-type AdaptiveAmassHdf5Dataset --inference-config.visualize-traj --inference-config.guidance-inner --inference-config.guidance-outer
   ```

   This will run inference and then post-process the results (with guidance optimization).

<!-- 3. **Running inference on your own data.**

   To run inference on your own data, you can copy the structure of the example trajectories. The key files are:

   - A VRS file from Project Aria, which contains calibrations and images.
   - SLAM outputs from Project Aria's MPS: `closed_loop_trajectory.csv` and `semidense_points.csv.gz`.
   - (optional) HaMeR outputs, which we save to a `hamer_outputs.pkl`.
   - (optional) Project Aria wrist and palm tracking outputs.

4. **Running HaMeR on your own data.**

   To generate the `hamer_outputs.pkl` file, you'll need to install [hamer_helper](https://github.com/brentyi/hamer_helper).

   Then, as an example for running on our coffeemachine sequence:

   ```bash
   python 2_run_hamer_on_vrs.py --traj-root ./egoallo_example_trajectories/coffeemachine
   ``` -->

<!-- ## Status -->
<!-- 
This repository currently contains:

- `egoallo` package, which contains reference training and sampling implementation details.
- Training script.
- Model checkpoints.
- Dataset preprocessing script.
- Inference script.
- Visualization script.
- Setup instructions.

While we've put effort into cleaning up our code for release, this is research
code and there's room for improvement. If you have questions or comments,
please reach out! -->