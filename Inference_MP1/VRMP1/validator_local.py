"""
validator_local.py — Robust student self-check script.

Place this inside your VRMP1_<roll_number>/ folder and run:

    python validator_local.py

This validates with 100% coverage:
  ✓ All required files and weights exist
  ✓ predictor.py imports without errors
  ✓ CLS_CLASS_MAPPING and SEG_CLASS_MAPPING are correctly filled
  ✓ All 4 functions are implemented (not NotImplementedError)
  ✓ Models load successfully
  ✓ Classification output format is correct on a REAL image
  ✓ Detection + segmentation output format is correct on a REAL image
  ✓ Mask dimensions match the original image
  ✓ All value ranges and types are correct
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score

# DeepFashion2 category_id → name  (dataset constant)
DEEPFASHION_CATID_TO_NAME: Dict[int, str] = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}

# Populated at runtime from the student's CLS_CLASS_MAPPING
CANONICAL_CLASSES: set = set()
CANONICAL_CLASSES_LIST: list = []
NUM_CLASSES: int = 0
CANONICAL_NAME_TO_IDX: Dict[str, int] = {}
CATEGORY_ID_TO_CANONICAL: Dict[int, int] = {}

# ─── Counters ─────────────────────────────────────────────────────
_pass_count = 0
_fail_count = 0
_warn_count = 0


def _pass(msg: str):
    global _pass_count
    _pass_count += 1
    print(f"  [PASS] {msg}")


def _fail(msg: str):
    global _fail_count
    _fail_count += 1
    print(f"  [FAIL] {msg}")


def _warn(msg: str):
    global _warn_count
    _warn_count += 1
    print(f"  [WARN] {msg}")


def _check(condition: bool, pass_msg: str, fail_msg: str) -> bool:
    if condition:
        _pass(pass_msg)
        return True
    else:
        _fail(fail_msg)
        return False


# ─── Locate the real test image ──────────────────────────────────

def _find_test_image(folder: Path) -> Path | None:
    """Walk up from the student folder to find hidden_dataset/images/000001.jpg."""
    search = folder.parent  # workspace root (one level up from VRMP1_*)
    candidate = search / "hidden_dataset" / "images" / "000001.jpg"
    if candidate.exists():
        return candidate
    # Try any image in hidden_dataset
    hd = search / "hidden_dataset" / "images"
    if hd.is_dir():
        imgs = sorted(hd.glob("*.jpg"))
        if imgs:
            return imgs[0]
    return None


def _find_test_annotation(img_path: Path) -> Path | None:
    """Find the annotation JSON matching the test image."""
    annos_dir = img_path.parent.parent / "annos"
    anno_path = annos_dir / (img_path.stem + ".json")
    return anno_path if anno_path.exists() else None


# ─── GT loading & metric helpers ─────────────────────────────────

def load_annotation(anno_path: Path) -> List[Dict[str, Any]]:
    """Parse annotation JSON → list of GT items (only canonical classes)."""
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for val in data.values():
        if not isinstance(val, dict) or "bounding_box" not in val:
            continue
        cat_id = val["category_id"]
        if cat_id not in CATEGORY_ID_TO_CANONICAL:
            continue
        items.append({
            "box": val["bounding_box"],
            "segmentation": val["segmentation"],
            "category_id": cat_id,
            "category_name": val.get("category_name", ""),
            "canonical_idx": CATEGORY_ID_TO_CANONICAL[cat_id],
        })
    return items


def rasterize_polygons(segmentation: list, width: int, height: int) -> np.ndarray:
    """Render polygon coordinate lists into a binary (H, W) mask."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in segmentation:
        coords = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        if len(coords) >= 3:
            draw.polygon(coords, fill=1)
    return np.array(canvas, dtype=np.uint8)


def build_remap(student_mapping: dict) -> Dict[int, int]:
    """Map student class index → canonical class index by name matching."""
    remap: Dict[int, int] = {}
    for s_idx, s_name in student_mapping.items():
        name = str(s_name).strip().lower()
        if name in CANONICAL_NAME_TO_IDX:
            remap[int(s_idx)] = CANONICAL_NAME_TO_IDX[name]
    return remap


# ─── Class mapping validation ────────────────────────────────────

def validate_class_mapping(mapping, label: str, allow_background: bool) -> bool:
    if not _check(isinstance(mapping, dict),
                  f"{label} is a dict.",
                  f"{label} must be a dict, got {type(mapping).__name__}."):
        return False

    if not _check(len(mapping) > 0,
                  f"{label} is non-empty ({len(mapping)} entries).",
                  f"{label} is empty — you must fill in your class mapping!"):
        return False

    # Check keys are ints
    all_int_keys = all(isinstance(k, int) for k in mapping.keys())
    _check(all_int_keys,
           f"{label} keys are all integers.",
           f"{label} keys must be integers. Got: {[type(k).__name__ for k in mapping.keys()]}")

    # Check values are strings
    all_str_vals = all(isinstance(v, str) for v in mapping.values())
    _check(all_str_vals,
           f"{label} values are all strings.",
           f"{label} values must be strings.")

    # Check canonical class coverage
    clothing_names = set()
    for k, v in mapping.items():
        name = str(v).strip().lower()
        if name == "background":
            if not allow_background:
                _warn(f"{label}: index {k} is 'background' — not expected in CLS_CLASS_MAPPING.")
            continue
        clothing_names.add(name)

    missing = CANONICAL_CLASSES - clothing_names
    extra = clothing_names - CANONICAL_CLASSES
    if extra:
        _warn(f"{label}: unrecognized classes (will be ignored by evaluator): {extra}")

    if not _check(len(missing) == 0,
                  f"{label} covers all 5 canonical classes.",
                  f"{label} missing canonical classes: {missing}"):
        return False

    if not allow_background:
        expected = 5
        _check(len(mapping) == expected,
               f"{label} has exactly {expected} entries (no background).",
               f"{label} should have {expected} entries for classification, got {len(mapping)}.")
    return True


# ─── Classification output validation ────────────────────────────

def validate_cls_output(outputs: list, num_images: int, num_classes: int) -> bool:
    ok = True
    if not _check(isinstance(outputs, list),
                  "Classification returns a list.",
                  f"Classification must return a list, got {type(outputs).__name__}."):
        return False

    if not _check(len(outputs) == num_images,
                  f"Classification returned {num_images} result(s) for {num_images} image(s).",
                  f"Expected {num_images} results, got {len(outputs)}."):
        return False

    for idx, out in enumerate(outputs):
        prefix = f"cls_output[{idx}]"
        if not _check(isinstance(out, dict),
                      f"{prefix} is a dict.",
                      f"{prefix} must be a dict, got {type(out).__name__}."):
            ok = False
            continue

        if not _check("labels" in out,
                      f"{prefix} has 'labels' key.",
                      f"{prefix} missing 'labels' key. Keys found: {list(out.keys())}"):
            ok = False
            continue

        labels = out["labels"]
        if not _check(isinstance(labels, list),
                      f"{prefix}['labels'] is a list.",
                      f"{prefix}['labels'] must be a list, got {type(labels).__name__}."):
            ok = False
            continue

        if not _check(len(labels) == num_classes,
                      f"{prefix}['labels'] has length {num_classes}.",
                      f"{prefix}['labels'] must have length {num_classes}, got {len(labels)}."):
            ok = False
            continue

        all_valid = True
        for i, l in enumerate(labels):
            if not isinstance(l, int):
                _fail(f"{prefix}['labels'][{i}] must be int, got {type(l).__name__}.")
                ok = False
                all_valid = False
            elif l not in (0, 1):
                _fail(f"{prefix}['labels'][{i}] must be 0 or 1, got {l}.")
                ok = False
                all_valid = False

        if all_valid:
            _pass(f"{prefix}: all label values are valid (binary 0/1). Output: {labels}")
    return ok


# ─── Detection output validation ────────────────────────────────

def validate_det_output(outputs: list, num_images: int, img_sizes: list, max_label: int) -> bool:
    ok = True
    if not _check(isinstance(outputs, list),
                  "Detection returns a list.",
                  f"Detection must return a list, got {type(outputs).__name__}."):
        return False

    if not _check(len(outputs) == num_images,
                  f"Detection returned {num_images} result(s) for {num_images} image(s).",
                  f"Expected {num_images} results, got {len(outputs)}."):
        return False

    for idx, out in enumerate(outputs):
        w, h = img_sizes[idx]
        prefix = f"det_output[{idx}]"

        if not _check(isinstance(out, dict),
                      f"{prefix} is a dict.",
                      f"{prefix} must be a dict."):
            ok = False
            continue

        required_keys = {"boxes", "scores", "labels", "masks"}
        present_keys = set(out.keys())
        missing_keys = required_keys - present_keys
        if not _check(len(missing_keys) == 0,
                      f"{prefix} has all required keys (boxes, scores, labels, masks).",
                      f"{prefix} missing keys: {missing_keys}"):
            ok = False
            continue

        n = len(out["boxes"])
        lengths_ok = (len(out["scores"]) == n and len(out["labels"]) == n
                      and len(out["masks"]) == n)
        if not _check(lengths_ok,
                      f"{prefix}: all arrays have same length ({n} detections).",
                      f"{prefix}: length mismatch — boxes={n}, scores={len(out['scores'])}, "
                      f"labels={len(out['labels'])}, masks={len(out['masks'])}."):
            ok = False
            continue

        if n == 0:
            _warn(f"{prefix}: zero detections — model may be undertrained or image has no objects.")
            continue

        # Boxes
        boxes_valid = True
        for i, box in enumerate(out["boxes"]):
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                _fail(f"{prefix}/boxes[{i}] must be [x1,y1,x2,y2].")
                ok = False
                boxes_valid = False
            else:
                x1, y1, x2, y2 = [float(c) for c in box]
                if not (x1 < x2 and y1 < y2):
                    _fail(f"{prefix}/boxes[{i}]: need x1<x2 and y1<y2, got [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}].")
                    ok = False
                    boxes_valid = False
        if boxes_valid:
            _pass(f"{prefix}: all {n} boxes have valid [x1,y1,x2,y2] format.")

        # Scores
        scores_valid = True
        for i, s in enumerate(out["scores"]):
            if not isinstance(s, (int, float)):
                _fail(f"{prefix}/scores[{i}] must be numeric, got {type(s).__name__}.")
                ok = False
                scores_valid = False
            elif not (0.0 <= float(s) <= 1.0):
                _fail(f"{prefix}/scores[{i}] must be in [0,1], got {s}.")
                ok = False
                scores_valid = False
        if scores_valid:
            _pass(f"{prefix}: all {n} scores in [0, 1].")

        # Labels
        labels_valid = True
        for i, l in enumerate(out["labels"]):
            if not isinstance(l, int):
                _fail(f"{prefix}/labels[{i}] must be int, got {type(l).__name__}.")
                ok = False
                labels_valid = False
            elif not (0 <= l <= max_label):
                _fail(f"{prefix}/labels[{i}] must be in [0, {max_label}], got {l}.")
                ok = False
                labels_valid = False
        if labels_valid:
            _pass(f"{prefix}: all {n} labels are valid integers in [0, {max_label}].")

        # Masks
        masks_valid = True
        for i, mask in enumerate(out["masks"]):
            arr = np.asarray(mask)
            if arr.ndim != 2:
                _fail(f"{prefix}/masks[{i}] must be 2D, got {arr.ndim}D shape={arr.shape}.")
                ok = False
                masks_valid = False
                continue
            if arr.shape != (h, w):
                _fail(f"{prefix}/masks[{i}] shape {arr.shape} != image size ({h}, {w}). "
                      "You must resize masks back to the original image resolution!")
                ok = False
                masks_valid = False
            uniq = set(np.unique(arr).tolist())
            if not uniq.issubset({0, 1}):
                _fail(f"{prefix}/masks[{i}] must be binary (0/1), got values {uniq}.")
                ok = False
                masks_valid = False
        if masks_valid and n > 0:
            _pass(f"{prefix}: all {n} masks are binary and match image size ({h}x{w}).")

    return ok


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    folder = Path(__file__).resolve().parent
    print("=" * 60)
    print(f"  VALIDATOR — {folder.name}")
    print("=" * 60)

    # ─── 1. Required files ────────────────────────────────────────
    print("\n[1/6] Checking required files ...")
    abort = False
    if not _check((folder / "predictor.py").exists(),
                  "predictor.py found.",
                  "predictor.py NOT found!"):
        abort = True

    has_cls_weights = (folder / "model_files" / "cls.pt").exists() or \
                      (folder / "model_files" / "cls.pth").exists()
    has_seg_weights = (folder / "model_files" / "seg.pt").exists() or \
                      (folder / "model_files" / "seg.pth").exists()

    if has_cls_weights:
        _pass("model_files/cls.pt(h) found.")
    else:
        _warn("model_files/cls.pt(h) not found — OK if classification reuses the seg model.")

    if not _check(has_seg_weights,
                  "model_files/seg.pt(h) found.",
                  "model_files/seg.pt(h) NOT found!"):
        abort = True

    if abort:
        print("\n[ABORT] Fix missing files before continuing.")
        sys.exit(1)

    # ─── 2. Find test image + annotation ──────────────────────────
    print("\n[2/6] Locating test image ...")
    test_img_path = _find_test_image(folder)
    if test_img_path is None:
        _fail("Cannot find hidden_dataset/images/000001.jpg — "
              "make sure hidden_dataset/ is in the parent directory.")
        sys.exit(1)
    else:
        test_img = Image.open(test_img_path).convert("RGB")
        img_w, img_h = test_img.size
        _pass(f"Using real test image: {test_img_path.name} ({img_w}x{img_h})")

    anno_path = _find_test_annotation(test_img_path)

    # ─── 3. Import predictor ─────────────────────────────────────
    print("\n[3/6] Importing predictor.py ...")
    try:
        spec = importlib.util.spec_from_file_location("predictor", folder / "predictor.py")
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
        _pass("predictor.py imported successfully.")
    except Exception as e:
        _fail(f"predictor.py import error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ─── 4. Validate class mappings + function existence ─────────
    print("\n[4/6] Validating class mappings and function signatures ...")

    has_cls_map = hasattr(predictor, "CLS_CLASS_MAPPING")
    has_seg_map = hasattr(predictor, "SEG_CLASS_MAPPING")

    if not _check(has_cls_map,
                  "CLS_CLASS_MAPPING attribute exists.",
                  "CLS_CLASS_MAPPING not found in predictor.py!"):
        sys.exit(1)
    if not _check(has_seg_map,
                  "SEG_CLASS_MAPPING attribute exists.",
                  "SEG_CLASS_MAPPING not found in predictor.py!"):
        sys.exit(1)

    # Build canonical class structures from CLS_CLASS_MAPPING
    global CANONICAL_CLASSES, CANONICAL_CLASSES_LIST, NUM_CLASSES
    global CANONICAL_NAME_TO_IDX, CATEGORY_ID_TO_CANONICAL

    cls_names = []
    for idx in sorted(predictor.CLS_CLASS_MAPPING.keys()):
        name = str(predictor.CLS_CLASS_MAPPING[idx]).strip().lower()
        if name != "background":
            cls_names.append(name)
    CANONICAL_CLASSES_LIST = cls_names
    CANONICAL_CLASSES = set(cls_names)
    NUM_CLASSES = len(cls_names)
    CANONICAL_NAME_TO_IDX = {name: i for i, name in enumerate(cls_names)}

    CATEGORY_ID_TO_CANONICAL = {}
    for cat_id, cat_name in DEEPFASHION_CATID_TO_NAME.items():
        if cat_name in CANONICAL_NAME_TO_IDX:
            CATEGORY_ID_TO_CANONICAL[cat_id] = CANONICAL_NAME_TO_IDX[cat_name]

    _pass(f"Derived {NUM_CLASSES} canonical classes from CLS_CLASS_MAPPING: {cls_names}")

    validate_class_mapping(predictor.CLS_CLASS_MAPPING, "CLS_CLASS_MAPPING", allow_background=False)
    validate_class_mapping(predictor.SEG_CLASS_MAPPING, "SEG_CLASS_MAPPING", allow_background=True)

    # Load GT annotation now that canonical mapping is ready
    gt_items: List[Dict[str, Any]] = []
    if anno_path is not None:
        gt_items = load_annotation(anno_path)
        _pass(f"Loaded GT annotation: {anno_path.name} ({len(gt_items)} objects)")
    else:
        _warn("No annotation found — metrics (F1, mIoU) will be skipped.")

    max_label = max(int(k) for k in predictor.SEG_CLASS_MAPPING.keys()) if predictor.SEG_CLASS_MAPPING else 5
    num_cls_classes = len(predictor.CLS_CLASS_MAPPING)

    # Check all 4 required functions exist and are callable
    required_fns = [
        "load_classification_model",
        "predict_classification",
        "load_detection_model",
        "predict_detection_segmentation",
    ]
    for fn_name in required_fns:
        if not _check(hasattr(predictor, fn_name) and callable(getattr(predictor, fn_name)),
                      f"{fn_name}() exists and is callable.",
                      f"{fn_name}() NOT found or not callable!"):
            sys.exit(1)

    # ─── 5. Test classification pipeline ─────────────────────────
    print(f"\n[5/6] Testing classification on real image ({img_w}x{img_h}) ...")
    device = "cpu"

    # 5a. load_classification_model — must NOT raise NotImplementedError
    cls_model = None
    try:
        cls_model = predictor.load_classification_model(str(folder), device)
        _pass("load_classification_model() returned successfully.")
    except NotImplementedError:
        _fail("load_classification_model() raises NotImplementedError — "
              "you MUST implement this function!")
    except Exception as e:
        _fail(f"load_classification_model() raised: {e}")
        traceback.print_exc()

    # 5b. predict_classification — must NOT raise NotImplementedError
    cls_out = None
    if cls_model is not None:
        try:
            cls_out = predictor.predict_classification(cls_model, [test_img])
            _pass("predict_classification() returned successfully.")
            validate_cls_output(cls_out, num_images=1, num_classes=num_cls_classes)
        except NotImplementedError:
            _fail("predict_classification() raises NotImplementedError — "
                  "you MUST implement this function!")
            cls_out = None
        except Exception as e:
            _fail(f"predict_classification() raised: {e}")
            traceback.print_exc()
            cls_out = None

    # 5c. Compute macro F1 if GT annotation is available
    if cls_out is not None and anno_path is not None:
        try:
            remap_cls = build_remap(predictor.CLS_CLASS_MAPPING)
            gt_vec = np.zeros(NUM_CLASSES, dtype=np.int32)
            for item in gt_items:
                gt_vec[item["canonical_idx"]] = 1
            pred_vec = np.zeros(NUM_CLASSES, dtype=np.int32)
            student_labels = cls_out[0]["labels"]
            for s_idx, val in enumerate(student_labels):
                canonical = remap_cls.get(s_idx)
                if canonical is not None:
                    pred_vec[canonical] = val
            macro_f1 = float(f1_score(
                gt_vec.reshape(1, -1), pred_vec.reshape(1, -1),
                average="macro", zero_division=0.0,
            ))
            print(f"\n  ** Classification Macro F1: {macro_f1:.4f} **")
        except Exception as e:
            _warn(f"Could not compute macro F1: {e}")

    # ─── 6. Test detection + segmentation pipeline ───────────────
    print(f"\n[6/6] Testing detection + segmentation on real image ({img_w}x{img_h}) ...")

    # 6a. load_detection_model — must NOT raise NotImplementedError
    det_model = None
    try:
        det_model = predictor.load_detection_model(str(folder), device)
        _pass("load_detection_model() returned successfully.")
    except NotImplementedError:
        _fail("load_detection_model() raises NotImplementedError — "
              "you MUST implement this function!")
    except Exception as e:
        _fail(f"load_detection_model() raised: {e}")
        traceback.print_exc()

    # 6b. predict_detection_segmentation — must NOT raise NotImplementedError
    det_out = None
    if det_model is not None:
        try:
            det_out = predictor.predict_detection_segmentation(det_model, [test_img])
            _pass("predict_detection_segmentation() returned successfully.")
            validate_det_output(
                det_out,
                num_images=1,
                img_sizes=[(img_w, img_h)],
                max_label=max_label,
            )
        except NotImplementedError:
            _fail("predict_detection_segmentation() raises NotImplementedError — "
                  "you MUST implement this function!")
            det_out = None
        except Exception as e:
            _fail(f"predict_detection_segmentation() raised: {e}")
            traceback.print_exc()
            det_out = None

    # 6c. Compute mIoU if GT annotation is available
    if det_out is not None and anno_path is not None and len(det_out) > 0:
        try:
            remap_seg = build_remap(predictor.SEG_CLASS_MAPPING)
            pred = det_out[0]
            IGNORE_LABEL = 255

            # Build predicted semantic map (highest-confidence per pixel)
            pred_sem = np.full((img_h, img_w), IGNORE_LABEL, dtype=np.uint8)
            pred_conf = np.full((img_h, img_w), -1.0, dtype=np.float32)
            for mask, score, label in zip(
                pred["masks"], pred["scores"], pred["labels"]
            ):
                canonical = remap_seg.get(label)
                if canonical is None:
                    continue
                binary = np.asarray(mask, dtype=np.uint8)
                if binary.shape != (img_h, img_w):
                    mask_pil = Image.fromarray(binary * 255)
                    mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
                    binary = (np.array(mask_pil) > 127).astype(np.uint8)
                higher = (binary == 1) & (score > pred_conf)
                pred_sem[higher] = canonical
                pred_conf[higher] = score

            # Build GT semantic map from polygon annotations
            gt_sem = np.full((img_h, img_w), IGNORE_LABEL, dtype=np.uint8)
            for item in gt_items:
                gt_mask = rasterize_polygons(item["segmentation"], img_w, img_h)
                gt_sem[gt_mask == 1] = item["canonical_idx"]

            # Per-class IoU
            intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
            union = np.zeros(NUM_CLASSES, dtype=np.float64)
            for c in range(NUM_CLASSES):
                pred_c = (pred_sem == c)
                gt_c = (gt_sem == c)
                intersection[c] = np.logical_and(pred_c, gt_c).sum()
                union[c] = np.logical_or(pred_c, gt_c).sum()

            per_class_iou = []
            for c in range(NUM_CLASSES):
                if union[c] > 0:
                    per_class_iou.append(float(intersection[c] / union[c]))
                else:
                    per_class_iou.append(float("nan"))

            valid_ious = [v for v in per_class_iou if not np.isnan(v)]
            miou = float(np.mean(valid_ious)) if valid_ious else 0.0

            print(f"\n  ** Segmentation mIoU: {miou:.4f} **")
            for c in range(NUM_CLASSES):
                iou_str = f"{per_class_iou[c]:.4f}" if not np.isnan(per_class_iou[c]) else "N/A"
                print(f"       {CANONICAL_CLASSES_LIST[c]:20s}: {iou_str}")
        except Exception as e:
            _warn(f"Could not compute mIoU: {e}")

    # ─── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  RESULTS:  {_pass_count} passed, {_fail_count} failed, "
          f"{_warn_count} warnings")
    print("=" * 60)
    if _fail_count > 0:
        print("\n  VALIDATION FAILED — fix the [FAIL] items above before submitting.\n")
        sys.exit(1)
    elif _warn_count > 0:
        print("\n  VALIDATION PASSED WITH WARNINGS — review [WARN] items above.\n")
    else:
        print("\n  ALL CHECKS PASSED — your submission looks good!\n")


if __name__ == "__main__":
    main()
