"""Limb definitions and conversion of keypoints to skeletons and heatmaps."""

import torch


BODY_LIMBS = [(0, 5), (0, 6), (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
              (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16)]

# Preserve the 20 custom face limbs and the 83-channel ordering in the supplied notebooks.
FACE_LIMBS = ([(36, 41), (37, 40), (38, 39), (42, 47), (43, 46), (44, 45),
               (17, 19), (19, 21), (22, 24), (24, 26)]
              + [(i, i + 1) for i in range(60, 67)] + [(67, 60), (51, 56), (50, 57)])
FACE_CONTOURS = ([(i, i + 1) for i in range(16)]
                 + [(i, i + 1) for i in range(17, 21)]
                 + [(i, i + 1) for i in range(22, 26)]
                 + [(i, i + 1) for i in range(27, 30)]
                 + [(i, i + 1) for i in range(31, 35)]
                 + [(i, i + 1) for i in range(36, 41)] + [(41, 36)]
                 + [(i, i + 1) for i in range(42, 47)] + [(47, 42)]
                 + [(i, i + 1) for i in range(48, 59)] + [(59, 48)]
                 + [(i, i + 1) for i in range(60, 67)] + [(67, 60)])
FACE_HEATMAP_LIMBS = FACE_LIMBS + FACE_CONTOURS


def valid_keypoints(points, keypoints, image_size, threshold):
    """Return (people,K,2) pixel coordinates; -1 marks invalid or low-score joints."""
    values = torch.as_tensor(points, dtype=torch.float32)
    if values.numel() == 0:
        return torch.empty(0, keypoints, 2)
    if values.ndim != 3 or values.shape[1:] != (keypoints, 3) or not torch.isfinite(values).all():
        raise ValueError(f"Expected finite (people,{keypoints},3) x,y,score annotations.")
    xy = values[:, :, :2].clone()
    visible = (values[:, :, 2] >= threshold) & (xy >= 0).all(-1) & (xy < image_size).all(-1)
    xy[~visible] = -1
    return xy


def skeleton_targets(points, limbs, layout="channels_first"):
    """Limb endpoints plus a limb graph: distinct limbs sharing a joint are adjacent."""
    pairs = torch.tensor(limbs, dtype=torch.long)
    coordinates = points[:, pairs].reshape(len(points), len(limbs), 4)
    visible = (coordinates >= 0).all(-1)
    graph = torch.tensor([[i != j and bool(set(a) & set(b)) for j, b in enumerate(limbs)]
                          for i, a in enumerate(limbs)], dtype=torch.float32)
    adjacency = graph[None] * visible[:, :, None] * visible[:, None, :]
    if layout == "channels_first":
        coordinates = coordinates.transpose(1, 2)
    elif layout != "limb_first":
        raise ValueError("Unknown coordinate layout.")
    return coordinates.reshape(len(points), 4 * len(limbs)).contiguous(), adjacency


def heatmap_targets(points, limbs, image_size, heatmap_size=56, sigma=1.0):
    """Sum Gaussian distances to visible limb segments, then clamp each channel to 1."""
    if sigma <= 0:
        raise ValueError("Heatmap sigma must be positive.")
    heatmaps = torch.zeros(len(limbs), heatmap_size, heatmap_size)
    yy, xx = torch.meshgrid(torch.arange(heatmap_size, dtype=torch.float32),
                            torch.arange(heatmap_size, dtype=torch.float32), indexing="ij")
    for channel, (start, end) in enumerate(limbs):
        for person in points:
            segment = person[[start, end]]
            if (segment < 0).any():
                continue
            (x1, y1), (x2, y2) = segment * (heatmap_size / image_size)
            dx, dy = x2 - x1, y2 - y1
            length = dx.square() + dy.square()
            if length < 1e-6:
                distance = (xx - x1).square() + (yy - y1).square()
            else:
                fraction = (((xx - x1) * dx + (yy - y1) * dy) / length).clamp(0, 1)
                distance = (xx - x1 - fraction * dx).square() + (yy - y1 - fraction * dy).square()
            heatmaps[channel] += torch.exp(-distance / (2 * sigma**2))
    return heatmaps.clamp_max(1)
