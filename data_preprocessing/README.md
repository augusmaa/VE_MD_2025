# VE-MD preprocessing

Offline preparation of frames, cached ViT features and structural supervision
for **VE-MD-PersonQuery** and **VE-MD-Heatmap**. Keep this directory separate
from the two training repositories. Both training variants read the same frames
and features, with different structural targets.

| File | Purpose |
|---|---|
| `extract_frames.py` | Sample RGB frames from videos, images or existing HDF5 data |
| `extract_features.py` | Save one ViT vector per frame |
| `extract_body.py` | Run ViTPose-ONNX for COCO-17 body keypoints |
| `extract_faces.py` | Run Face Alignment for 68 facial landmarks |
| `prepare_targets.py` | Save skeleton/adjacency tensors and dense heatmaps |
| `make_manifest.py` | Write the data lists accepted by both training repositories |
| `structures.py` | Limb definitions, adjacency and Gaussian heatmap construction |
| `data.py` | Shared frame loading, normalization and file utilities |

## Installation

```bash
conda env create -f environment.yml
conda activate ve-md-preprocessing
```

Body extraction uses the same external
[ViTPose-ONNX implementation](https://github.com/Pukei-Pukei/ViTPose-ONNX)
as the supplied scripts. Obtain its YOLOv6 and COCO-17 ViTPose ONNX weights
from that repository's download instructions. The adapter was written against
revision `aa5542fab538179d1c7866b8f2d4fd3c12e3339a`:

```bash
git clone https://github.com/Pukei-Pukei/ViTPose-ONNX.git external/ViTPose-ONNX
git -C external/ViTPose-ONNX checkout aa5542fab538179d1c7866b8f2d4fd3c12e3339a
```

The environment includes CPU ONNX Runtime. For CUDA body extraction, use the
matching `onnxruntime-gpu==1.20.1` package instead of `onnxruntime`, with its
[CUDA dependencies](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
ViT extraction accepts CPU and CUDA; body and face extraction default to
CPU. [Face Alignment](https://github.com/1adrianb/face-alignment)
downloads its detector/landmark weights on first use. These external weights
are not bundled here.

## 1. Prepare and sample a split

Create a CSV with one row per example:

```csv
id,path,label
clip001,/datasets/VGAF/Train/clip001.mp4,0
clip002,/datasets/VGAF/Train/clip002.mp4,2
```

`id` must be unique within the split. Paths may be absolute or relative to the
CSV. Labels must already be integers from 0 to `num_classes - 1`, with the same
class order in every split. Apply the dataset's label mapping when preparing
this CSV; none of the extraction stages remaps labels. Use the official splits.
For an existing HDF5 source, add a `key` column identifying each RGB frame array.

```bash
python extract_frames.py --samples train.csv --frames 5 --size 224 --output prepared/train
python extract_frames.py --samples val.csv --frames 5 --size 224 --output prepared/val
```

Use `--frames 1` for image datasets such as GAF-3.0. Set the frame count to the
experiment's sampling protocol. Video sampling uses FFmpeg's full-clip FPS
filter, as in the supplied extraction script. Short decoded sequences repeat
their final frame; undecodable videos raise an error. Existing HDF5 sequences
use evenly spaced frame indices. The selected indices or FPS policy are saved.

Each split contains `frames.h5` with uint8 RGB arrays `(T,H,W,3)` and a
`frames.json` index. All later stages use exactly these saved frames and their
order. Resize is deterministic; no independent random crop or flip is applied.

## 2. Extract ViT features

Pass the **exact checkpoint** used for the experiment:

```bash
python extract_features.py --index prepared/train/frames.json \
  --checkpoint checkpoints/vit_vgaf.tar --output prepared/train/features
python extract_features.py --index prepared/val/frames.json \
  --checkpoint checkpoints/vit_vgaf.tar --output prepared/val/features
```

The default backbone is `vit_large_patch14_clip_224.laion2b`, the current timm
name for the original `vit_large_patch14_224_clip_laion2b`. The loader accepts
the supplied `VideoVitNet` checkpoints (`model_state_dict`, including a
`module.vit.` prefix), or a backbone state dictionary. Pooling/classifier weights
from the video wrapper are excluded; backbone loading is strict.

Outputs are CPU float tensors `(T,1024)`, with no video-batch dimension and no
temporal pooling. The backbone is frozen and runs in evaluation mode.
The ImageNet normalization from the original extraction code is the default.
Use `--normalization model` only when the chosen weights require their timm
mean/std; this option changes normalization, not frame geometry. `--pretrained`
explicitly selects published timm weights instead of an experiment checkpoint.
These weight choices are different experimental settings.

The feature metadata records the model, timm version, normalization, checkpoint
hash and source frame index. The implementation follows timm's
[pre-classifier feature API](https://huggingface.co/docs/timm/feature_extraction).

## 3. Extract training annotations

```bash
python extract_body.py --index prepared/train/frames.json \
  --repo external/ViTPose-ONNX --detector checkpoints/yolov6m.onnx \
  --pose checkpoints/vitpose-b-multi-coco.onnx --output prepared/train/body.json

python extract_faces.py --index prepared/train/frames.json \
  --output prepared/train/faces.json
```

Each backend loads once. Body inference processes one frame at a time and
converts ViTPose's `(y,x,score)` output to `(x,y,score)`. Facial landmarks are
already `(x,y)`; a score of 1 marks a detected landmark, not calibrated
confidence. Annotation files include every sample and frame. An empty list
means no detections. Body and face detections are independent; no person
tracking or body-to-face association is inferred.

## 4. Build both structural representations

```bash
python prepare_targets.py --index prepared/train/frames.json \
  --body prepared/train/body.json --faces prepared/train/faces.json \
  --output prepared/train/targets
```

Provide only `--body` or only `--faces` for single-branch supervision. The default
confidence threshold is 0.1. Out-of-image or low-confidence keypoints become
`-1`; coordinates on the top or left border are valid.

| Target per sample | Body | Face |
|---|---|---|
| PersonQuery coordinates, list of T tensors | `(N_t,72)` | `(N_t,80)` |
| PersonQuery adjacency, list of T tensors | `(N_t,18,18)` | `(N_t,20,20)` |
| Heatmap tensor | `(T,18,56,56)` | `(T,83,56,56)` |

Skeleton coordinates are pixels in the resized image. The default flattened
layout is `(4,K)` (`channels_first`), matching the refined PersonQuery config.
`--coordinate-layout limb_first` writes `(K,4)`; use that same setting in
PersonQuery if selected. A limb's values are `x_start,y_start,x_end,y_end`.
The adjacency graph connects **distinct limbs sharing a landmark** and has a
zero diagonal. Rows/columns for incompletely observed limbs are zero.

Variable person counts are retained. Empty frames have zero people; people
are never duplicated to fill a tensor. PersonQuery's loader handles padding.

Heatmaps preserve the supplied 18 body edges and the ordered 83 face edges
(20 custom edges followed by 63 contour edges). Each channel sums Gaussian
distances to the corresponding visible segments and is clamped to `[0,1]`.
An empty detection set produces zeros. The default sigma, in heatmap pixels,
is 1 for body and 0.5 for face; both are command-line options. The supplied GAF
notebook used the same sigma for body and face, so set `--face-sigma 1` for
that setting. Record the parameters used for each experiment.

## 5. Connect the training repositories

```bash
python make_manifest.py --index prepared/train/frames.json \
  --features prepared/train/features --targets prepared/train/targets --output prepared/train
python make_manifest.py --index prepared/val/frames.json \
  --features prepared/val/features --output prepared/val
```

This writes `personquery.json` and `heatmap.json` for each split. Validation
requires no body/face annotations. Set `data.train` and `data.validation` in
each training repository's `config.yaml` to the corresponding JSON files.
Use the same frame count, image size, class count and enabled structural
branches. The default ViT feature dimension is 1024. If features are omitted,
set `use_vit: false` in training.

The data lists are already in the training format; the training repositories'
`tools/prepare_data.py` adapter is unnecessary for these outputs. Paths within
the lists are relative, so the prepared directory can be moved as a whole.

Extraction directories/files must be new. Existing outputs are never deleted
or silently overwritten. Each stage records its source frame index and rejects
mixing different indices when creating targets/data lists. After changing
sampling, geometry, weights or annotation settings, create a new prepared
directory and regenerate the dependent outputs together. In particular, old
caches produced with dropped frames, independent crops or duplicated people
should not be mixed with this workflow.
