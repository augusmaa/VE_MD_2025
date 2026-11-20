# VE-MD-Heatmap

Training code for VE-MD with dense structural heatmap decoders. Each run trains
one classification task on one dataset. The ViT branch consumes features
extracted in advance, using the same interface and latent mapping as PersonQuery.

## Code

| File | Purpose |
|---|---|
| `run.py` | Configure the model/loaders, train, validate and save checkpoints |
| `train.py` | One complete training epoch |
| `validation.py` | Validation loss, accuracy and macro-F1 |
| `model.py` | Visual/cached-feature fusion and classification |
| `encoders.py` | Variational ResNet and custom residual encoders |
| `decoders.py` | Dense heatmap decoders and temporal positional encodings |
| `loss.py` | Classification, stage-wise heatmap MSE and MMD |
| `dataset.py` | Prepared frames, cached ViT features and heatmap targets |
| `config.yaml` | Data paths and experiment parameters |

## Installation and training

```bash
conda env create -f ve_md_env.yml
conda activate ve-md-heatmap
```

Set the training/validation file lists and parameters in `config.yaml`, then run:

```bash
python run.py --config config.yaml --output results/experiment
```

The supplied configuration is a starting point, not a claim to reproduce a
particular result table. `--device auto` selects CUDA or CPU. Set
`--device cpu` or `cuda` to choose explicitly.

For DDP on NVIDIA GPUs:

```bash
conda env create -f ve_md_env_cuda.yml
conda activate ve-md-heatmap-cuda
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py \
  --config config.yaml --device cuda --output results/experiment
```

GPU DDP uses the same launch with `--device cpu`.
The batch size is per process; only rank zero writes outputs.

## Prepared data and cached ViT features

Each split is a JSON list. Record paths are relative to the JSON file or absolute;
the split-list paths are relative to `config.yaml`. A training record is:

```json
[
  {
    "images": "train/clip01.mp4.pt",
    "label": 0,
    "feat_vit": "features/clip01.vit_feat.pt",
    "limb_person": "train/clip01.mp4_person_limb.pt",
    "limb_face": "train/clip01.mp4_face_limb.pt"
  }
]
```

Only enabled body/face branches require their targets. Validation needs no
heatmap targets. Labels are integers from 0 to `num_classes - 1`.

| Tensor per sample | Shape |
|---|---|
| Preprocessed video frames | `(T,3,H,W)` |
| Cached ViT features | `(T,D)`, normally D=1024 |
| Body heatmaps (`limb_person`) | `(T,18,56,56)` by default |
| Face heatmaps (`limb_face`) | `(T,83,56,56)` by default |

Heatmaps are dense, already aggregated over people. There is no person axis,
adjacency tensor, Hungarian matching or person-count padding. A single-image
heatmap `(C,56,56)` is accepted when T=1. Targets are loaded unchanged;
zero-valued background pixels participate in the MSE.

Each cached ViT vector must correspond to the same frame as the visual input.
Use the extraction weights and preprocessing associated with the experiment.
The training model does not instantiate ViT, read a ViT checkpoint or extract
features from images. It learns:

`(B,T,D) -> Linear(D,L*7*7) -> (B*T,L,7,7) -> mu/logvar -> latent`.


Video `.pt` files contain already-preprocessed floating-point tensors. Single
image files and uint8 HDF5 RGB arrays `(T,H,W,3)` are also accepted and are resized
and ImageNet-normalized. HDF5 input uses, for example,
`"images": {"path": "Val.h5", "key": "clip01.mp4.img"}`. Frame order must agree
across video, features and targets. Spatial augmentation is not applied separately
to these aligned inputs.

`tools/prepare_data.py` builds the JSON lists from the supplied prepared-file
conventions for VGAF, GAF-3.0, MER, DFEW, SamSemo and EngageNet. It keeps the
existing label mappings and `_person_limb.pt`/`_face_limb.pt` filenames, including
MER's `stem.img` training names. It references existing data; it does not extract
ViT features or generate annotations. Run `python tools/prepare_data.py --help`.

## Model and objective

The structural decoder upsamples 7 x 7 latent maps to 56 x 56 features. Six heads
share those features by default, preserving the supplied stage design. Each head
produces logits; the heatmap loss sums mean squared errors after sigmoid over all
stages. Body and face losses have separate weights in `config.yaml`.

The final head's sigmoid heatmaps are summed over channels before classification.
`add_heatmaps: true` enables this structural input. `projection_dim: null` uses
the flattened summed maps; an integer enables LayerNorm/linear/ReLU projection.
`add_heatmaps: false` retains auxiliary heatmap supervision only. `two_encoders`
enables separate body/face visual encoders. Temporal attention uses `(B,T,C)`
tensors consistently for both raw and projected structures.

The objective is cross-entropy plus weighted body/face heatmap losses and RBF
MMD over flattened frame latents. MMD is calculated within each process's batch.
Validation uses latent means unless `sample_in_eval` is enabled, as in the
PersonQuery implementation. ResNet ImageNet V1 initialization is controlled by
`pretrained`; it is independent of the externally cached ViT features.

## Metrics and checkpoints

Validation reports accuracy, macro-F1, per-class F1 and a confusion matrix.
Scores are stored on the 0–1 scale; console accuracy is a percentage. Macro-F1
averages over all configured classes, assigning zero to undefined class F1.
DDP reduces confusion counts before calculating metrics, and validation subsets
contain no duplicated samples. The training sampler can repeat a few samples
to equalize rank lengths.

Runs save `config.yaml`, `metrics.jsonl`, `last.pt` and `best.pt`. The best model
is selected by validation accuracy. To resume, keep the same experiment options,
increase `training.epochs` to the desired total and run:

```bash
python run.py --config config.yaml --resume results/experiment/last.pt \
  --output results/experiment
```

To evaluate:

```bash
python run.py --config config.yaml --evaluate --resume results/experiment/best.pt \
  --output results/evaluation
```
