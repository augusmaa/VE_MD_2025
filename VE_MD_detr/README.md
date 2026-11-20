# VE-MD-PersonQuery

Training code for VE-MD with PersonQuery structural decoders. Each run trains
one classification task on one dataset. Body and face structures provide
auxiliary supervision; the classifier can also use the predicted structures.

## Code

| File | Purpose |
|---|---|
| `run.py` | Load the configuration, create the model/loaders, train and validate |
| `train.py` | One training epoch |
| `validation.py` | Validation loss, accuracy and macro-F1 |
| `model.py` | VE-MD architecture and classification head |
| `encoders.py` | Variational visual encoders |
| `decoders.py` | PersonQuery, positional encodings and ST-GCN |
| `loss.py` | Classification, structural matching and MMD losses |
| `dataset.py` | Load prepared inputs and batch variable person counts |
| `config.yaml` | Data paths and experiment parameters |

## Installation and training

```bash
conda env create -f ve_md_env.yml
conda activate ve-md-personquery
```

Set the training and validation file lists in `config.yaml`, then run:

```bash
python run.py --config config.yaml --output results/experiment
```

The supplied configuration is a starting configuration, not a claim to reproduce
a particular result table. Set the parameters and data split for the experiment.
`device=auto` selects CUDA or CPU; `--device` can select one explicitly.

For multiple NVIDIA GPUs, install `ve_md_env_cuda.yml`, then launch with DDP:

```bash
conda env create -f ve_md_env_cuda.yml
conda activate ve-md-personquery-cuda
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py \
  --config config.yaml --device cuda --output results/experiment
```

DDP uses the same command with `--device cpu`. The configured batch size is per process. Only rank zero writes output files.

## Prepared data

Each split is a JSON list. Paths in each record are relative to that JSON file
or absolute. Dataset-list paths in `config.yaml` are relative to that config.
Training records have this form:

```json
[
  {
    "images": "frames/clip01.pt",
    "label": 0,
    "feat_vit": "features/clip01.pt",
    "skeleton_person": "structures/clip01_person_kp.pt",
    "adjacency_person": "structures/clip01_person_adj.pt"
  }
]
```

Add `skeleton_face` and `adjacency_face` when face supervision is enabled.
Validation needs only `images`, `label`, and `feat_vit` when the ViT branch is
enabled. Labels are integers from 0 to `num_classes - 1`.

| Input per sample | Shape |
|---|---|
| Preprocessed video tensor | `(T,3,H,W)` |
| Cached ViT features | `(T,D)` |
| Body/face coordinates | `(T,N,4*K)` |
| Body/face adjacency | `(T,N,K,K)` |

`T` is the frame count, `N` is the person count, and `K` is the number of
structural limbs: 18 for body and 20 for face by default. These are limb counts,
not heatmap landmark counts. Coordinates are pixels in the resized image plane;
-1 marks missing values. Empty/padded people are excluded from matching.
For different person counts within a clip, coordinate/adjacency files may hold
a list of T tensors. All inputs must share the same frame ordering.

Video `.pt` files contain already-preprocessed floating-point tensors. Single
image files and HDF5 RGB frame arrays `(T,H,W,3)` are also accepted; those inputs
are resized and ImageNet-normalized. An HDF5 `images` value is
`{"path": "Val.h5", "key": "clip01.mp4.img"}`. ViT feature extraction and
skeleton annotation happen before training.

`tools/prepare_data.py` creates these lists from the original prepared-file
conventions for VGAF, GAF-3.0, MER, DFEW, SamSemo and EngageNet. It preserves
their label mappings. See `python tools/prepare_data.py --help` for path arguments.
It does not extract or change the data.

## Model and objective

The model takes video frames and optional cached ViT features. Ground-truth
skeletons are loss targets. `add_keypoints` controls whether predicted
structures enter the classifier; `projection_dim: null` uses raw structures.
`two_encoders` separates body/face encoders, and `stgcn_active` enables graph and
temporal processing before classification. These options are set in `config.yaml`.
`coordinate_layout` must match the flattened coordinate order: `(4,K)` for
`channels_first`, `(K,4)` for `limb_first`.

The objective is cross-entropy plus weighted structural loss and MMD. Structural
loss uses Hungarian matching with masked Smooth L1 coordinates and adjacency
BCE. Matching supervises at most `min(num_queries, N)` people per frame. MMD uses
an RBF kernel on flattened frame latents, within each process's batch. Validation
uses latent means unless `sample_in_eval` is enabled. With `pretrained: true`,
ResNet uses torchvision ImageNet V1 weights. The optional `vitB` encoder requires
`pip install transformers`.

## Metrics and checkpoints

Validation reports accuracy and macro-F1 on the 0–1 scale; console accuracy is
shown as a percentage. Macro-F1 averages F1 over all configured classes, assigning
zero to an undefined class F1. DDP sums the confusion matrix before computing
metrics and evaluates each validation sample once. The standard training
sampler may repeat a few samples to equalize rank lengths.

Each run saves `config.yaml`, `metrics.jsonl`, `last.pt` and `best.pt`. The best
checkpoint is selected by validation accuracy. To resume, keep the same run
options, increase `training.epochs` to the desired total, and use:

```bash
python run.py --config config.yaml --resume results/experiment/last.pt \
  --output results/experiment
```

To evaluate a checkpoint without training:

```bash
python run.py --config config.yaml --evaluate --resume results/experiment/best.pt \
  --output results/evaluation
```
