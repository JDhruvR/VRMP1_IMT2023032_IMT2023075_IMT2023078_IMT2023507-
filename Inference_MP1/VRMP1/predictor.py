"""
predictor.py — Student inference file for hidden evaluation.

╔══════════════════════════════════════════════════════════════════╗
║  DO NOT RENAME ANY FUNCTION.                                    ║
║  DO NOT CHANGE FUNCTION SIGNATURES.                             ║
║  DO NOT REMOVE ANY FUNCTION.                                    ║
║  DO NOT RENAME CLS_CLASS_MAPPING or SEG_CLASS_MAPPING.          ║
║  You may add helper functions / imports as needed.              ║
╚══════════════════════════════════════════════════════════════════╝

Tasks
-----
  Task 3.1  — Multi-label image-level classification (5 classes).
  Task 3.2  — Object detection + instance segmentation (5 classes).

You must implement ALL FOUR functions below.

Class Mappings
--------------
  Fill in the two dictionaries below (CLS_CLASS_MAPPING, SEG_CLASS_MAPPING)
  to map your model's output indices to the canonical category names.

  The canonical 5 categories (from the DeepFashion2 subset) are:
      short sleeve top, long sleeve top, trousers, shorts, skirt

  Your indices can be in any order, but the category name strings
  must match exactly (case-insensitive). Background class is optional
  but recommended for detection/segmentation models — the evaluator
  will automatically ignore it.

  Important: Masks must be at the ORIGINAL image resolution.
  If your model internally resizes images, resize the masks back
  to the input image dimensions before returning them.

Model Weights
-------------
  Place your trained weights inside  model_files/  as:
      model_files/cls.pt   (or cls.pth)   — classification model
      model_files/seg.pt   (or seg.pth)   — detection + segmentation model

Evaluation Metrics
------------------
  Classification : Macro F1-score  +  Per-label macro accuracy
  Detection      : mAP @ [0.5 : 0.05 : 0.95]
  Segmentation   : Per-class mIoU (macro-averaged)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms  # ← this was missing
from torchvision.models import resnet50
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════
# CLASS MAPPINGS — FILL THESE IN
# ═══════════════════════════════════════════════════════════════════

# Classification: maps your model's output index → canonical class name.
# Must have exactly 5 entries (one per clothing class, NO background).
# Example:
#   CLS_CLASS_MAPPING = {
#       0: "short sleeve top",
#       1: "long sleeve top",
#       2: "trousers",
#       3: "shorts",
#       4: "skirt",
#   }
CLS_CLASS_MAPPING: Dict[int, str] = {
    0: "short sleeve top",
    1: "trousers",
    2: "shorts",
    3: "long sleeve top",
    4: "skirt",
}

# Detection + Segmentation: maps your model's output index → class name.
# Include background if your model outputs it (evaluator will ignore it).
# Example:
#   SEG_CLASS_MAPPING = {
#       0: "background",
#       1: "short sleeve top",
#       2: "long sleeve top",
#       3: "trousers",
#       4: "shorts",
#       5: "skirt",
#   }
SEG_CLASS_MAPPING: Dict[int, str] = {
    0: "short sleeve top",
    1: "long sleeve top",
    2: "shorts",
    3: "trousers",
    4: "skirt",
}


# ═══════════════════════════════════════════════════════════════════
# Helper utilities (you may modify or add more)
# ═══════════════════════════════════════════════════════════════════


def _find_weights(folder: Path, stem: str) -> Path:
    """Return the first existing weights file matching stem.pt or stem.pth."""
    for ext in (".pt", ".pth"):
        candidate = folder / "model_files" / (stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No weights file found for '{stem}' in {folder / 'model_files'}")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ImageNet normalization — same as training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_cls_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)

CLS_THRESHOLD = 0.5  # same threshold used in training evaluation

# ═══════════════════════════════════════════════════════════════════
# TASK 3.1 — CLASSIFICATION (ResNet-50)
# ═══════════════════════════════════════════════════════════════════


def load_classification_model(folder: str, device: str) -> Any:
    """Load ResNet-50 classification model from model_files/cls.pth"""
    folder = Path(folder)
    weights_path = _find_weights(folder, "cls")

    # Build same architecture as training
    model = resnet50(weights=None)
    model.fc = nn.Linear(2048, 5)  # 5 classes, no background

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Checkpoint was saved as dict with 'model_state_dict' key
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # fallback: assume raw state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {"model": model, "device": device}


def predict_classification(model: Any, images: List[Image.Image]) -> List[Dict]:
    """
    Multi-label classification with ResNet-50.
    Returns list of {"labels": [0/1, 0/1, 0/1, 0/1, 0/1]} per image.
    Order matches CLS_CLASS_MAPPING: short sleeve top, trousers, shorts, long sleeve top, skirt
    """
    net = model["model"]
    device = model["device"]

    results = []

    # Process in a single batch for efficiency
    batch = torch.stack([_cls_transform(img.convert("RGB")) for img in images]).to(device)  # (N, 3, 224, 224)

    with torch.no_grad():
        logits = net(batch)  # (N, 5)
        probs = torch.sigmoid(logits).cpu().numpy()  # (N, 5)

    for prob_row in probs:
        binary_labels = [int(p >= CLS_THRESHOLD) for p in prob_row]
        results.append({"labels": binary_labels})

    return results


# ═══════════════════════════════════════════════════════════════════
# TASK 3.2 — DETECTION + INSTANCE SEGMENTATION (YOLOv8-seg)
# ═══════════════════════════════════════════════════════════════════


def load_detection_model(folder: str, device: str) -> Any:
    """Load YOLOv8-seg model from model_files/seg.pt"""
    folder = Path(folder)
    weights_path = _find_weights(folder, "seg")

    model = YOLO(str(weights_path))

    # Move to correct device
    # YOLO uses device string directly: "cpu", "0", "cuda"
    if device == "cuda":
        yolo_device = "0"
    else:
        yolo_device = "cpu"

    return {"model": model, "device": yolo_device}


def predict_detection_segmentation(
    model: Any,
    images: List[Image.Image],
) -> List[Dict]:
    """
    Detection + instance segmentation with YOLOv8-seg.
    Returns list of {boxes, scores, labels, masks} per image.
    Masks are resized back to original image resolution.
    """
    net = model["model"]
    yolo_device = model["device"]

    results = []

    for img in images:
        img_rgb = img.convert("RGB")
        orig_w, orig_h = img_rgb.size  # PIL: (width, height)

        # Run YOLO inference
        # Pass PIL image directly — ultralytics handles it
        preds = net(
            img_rgb,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            device=yolo_device,
            verbose=False,
        )

        result = preds[0]  # single image result

        # ── Empty detection case ──────────────────────────────
        if result.boxes is None or len(result.boxes) == 0:
            results.append(
                {
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                    "masks": [],
                }
            )
            continue

        # ── Extract boxes, scores, labels ─────────────────────
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4) float32
        scores = result.boxes.conf.cpu().numpy()  # (N,)   float32
        labels = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        # Clip boxes to image bounds just in case
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        boxes_list = boxes_xyxy.tolist()  # list of [x1,y1,x2,y2]
        scores_list = scores.tolist()  # list of float
        labels_list = labels.tolist()  # list of int

        # ── Extract + resize masks ────────────────────────────
        masks_list = []

        if result.masks is not None:
            # result.masks.data shape: (N, mask_h, mask_w) — float32 in [0,1]
            raw_masks = result.masks.data.cpu().numpy()  # (N, mH, mW)

            for raw_mask in raw_masks:
                # Threshold to binary
                binary = (raw_mask > 0.5).astype(np.uint8)

                # Resize to original image size if needed
                if binary.shape != (orig_h, orig_w):
                    mask_pil = Image.fromarray(binary * 255, mode="L")
                    mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
                    binary = (np.array(mask_pil) > 127).astype(np.uint8)

                masks_list.append(binary)
        else:
            # YOLO returned boxes but no masks (shouldn't happen with seg model)
            # Fill with empty masks
            for _ in range(len(boxes_list)):
                masks_list.append(np.zeros((orig_h, orig_w), dtype=np.uint8))

        results.append(
            {
                "boxes": boxes_list,
                "scores": scores_list,
                "labels": labels_list,
                "masks": masks_list,
            }
        )

    return results


# # ═══════════════════════════════════════════════════════════════════
# # TASK 3.1 — CLASSIFICATION
# # ═══════════════════════════════════════════════════════════════════

# def load_classification_model(folder: str, device: str) -> Any:
#     """
#     Load your trained classification model.

#     Parameters
#     ----------
#     folder : str
#         Absolute path to your submission folder (the one containing
#         this predictor.py, model_files/, class_mapping_cls.json, etc.).
#     device : str
#         PyTorch device string, e.g. "cuda", "mps", or "cpu".

#     Returns
#     -------
#     model : Any
#         Whatever object your predict_classification function needs.
#         This is passed directly as the first argument to
#         predict_classification().

#     Notes
#     -----
#     - Load weights from  <folder>/model_files/cls.pt  (or .pth).
#     - Use CLS_CLASS_MAPPING defined above to map output indices.
#     - The returned object can be a dict, a nn.Module, or anything
#       your prediction function expects.
#     """
#     raise NotImplementedError("TODO: implement load_classification_model")


# def predict_classification(model: Any, images: List[Image.Image]) -> List[Dict]:
#     """
#     Run multi-label classification on a list of images.

#     Parameters
#     ----------
#     model : Any
#         The object returned by load_classification_model().
#     images : list of PIL.Image.Image
#         A list of RGB PIL images.

#     Returns
#     -------
#     results : list of dict
#         One dict per image, with the key "labels":

#         [
#             {"labels": [int, int, int, int, int]},
#             {"labels": [int, int, int, int, int]},
#             ...
#         ]

#         Each "labels" list has exactly 5 elements (one per class,
#         in the order defined by your CLS_CLASS_MAPPING dictionary).
#         Each element is 0 or 1.

#     Example
#     -------
#     >>> results = predict_classification(model, [img1, img2])
#     >>> results[0]
#     {"labels": [1, 0, 0, 1, 0]}
#     """
#     raise NotImplementedError("TODO: implement predict_classification")


# # ═══════════════════════════════════════════════════════════════════
# # TASK 3.2 — DETECTION + INSTANCE SEGMENTATION
# # ═══════════════════════════════════════════════════════════════════

# def load_detection_model(folder: str, device: str) -> Any:
#     """
#     Load your trained detection + segmentation model.

#     Parameters
#     ----------
#     folder : str
#         Absolute path to your submission folder.
#     device : str
#         PyTorch device string, e.g. "cuda", "mps", or "cpu".

#     Returns
#     -------
#     model : Any
#         Whatever object your predict_detection_segmentation function
#         needs. Passed directly as the first argument.

#     Notes
#     -----
#     - Load weights from  <folder>/model_files/seg.pt  (or .pth).
#     - Use SEG_CLASS_MAPPING defined above to map output indices.
#     """
#     raise NotImplementedError("TODO: implement load_detection_model")


# def predict_detection_segmentation(
#     model: Any,
#     images: List[Image.Image],
# ) -> List[Dict]:
#     """
#     Run detection + instance segmentation on a list of images.

#     Parameters
#     ----------
#     model : Any
#         The object returned by load_detection_model().
#     images : list of PIL.Image.Image
#         A list of RGB PIL images.

#     Returns
#     -------
#     results : list of dict
#         One dict per image with keys "boxes", "scores", "labels", "masks":

#         [
#             {
#                 "boxes":  [[x1, y1, x2, y2], ...],   # list of float coords
#                 "scores": [float, ...],               # confidence in [0, 1]
#                 "labels": [int, ...],                 # class indices (see mapping)
#                 "masks":  [np.ndarray, ...]           # binary masks, H×W, uint8
#             },
#             ...
#         ]

#     Output contract
#     ---------------
#     - boxes / scores / labels / masks must all have the same length
#       (= number of detected instances in that image).
#     - Each box is [x1, y1, x2, y2] with x1 < x2, y1 < y2.
#     - Coordinates must be within image bounds (0 ≤ x ≤ width, 0 ≤ y ≤ height).
#     - Each score is a float in [0, 1].
#     - Each label is an int index matching your SEG_CLASS_MAPPING.
#     - Each mask is a 2-D numpy array of shape (image_height, image_width)
#       with dtype uint8, containing only 0 and 1.
#     - If no objects are detected, return empty lists for all keys.

#     Example
#     -------
#     >>> results = predict_detection_segmentation(model, [img])
#     >>> results[0]["boxes"]
#     [[100.0, 40.0, 300.0, 420.0], [50.0, 200.0, 250.0, 600.0]]
#     >>> results[0]["masks"][0].shape
#     (height, width)
#     """
#     raise NotImplementedError("TODO: implement predict_detection_segmentation")
