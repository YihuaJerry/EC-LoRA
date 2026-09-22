#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


TASK_CONFIG_ALIASES = {
    "science_qa": "scienceqa",
    "image_net": "imagenet",
    "ocrvqa": "ocr_vqa",
    "ocr-vqa": "ocr_vqa",
    "grounding": "rec_coco",
    "rec-coco": "rec_coco",
    "aok_vqa": "aokvqa",
    "aok-vqa": "aokvqa",
    "imagenet-r": "imagenet_r",
    "imagenet_r": "imagenet_r",
    "screen_2_words": "screen2words",
    "screen-2-words": "screen2words",
    "tab_mwp": "tabmwp",
    "tab-mwp": "tabmwp",
}

MM_MERGEBENCH_ROOT = os.path.expanduser(os.environ.get("MM_MERGEBENCH_ROOT", "data/MM-MergeBench"))

UNSEEN_EVAL_SPECS: Dict[str, Dict[str, Any]] = {
    "aokvqa": {
        "config_task": "iconqa",
        "task": "iconqa",
        "question_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "AOKVQA", "val.json"),
        "annotation_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "AOKVQA", "val.json"),
        "image_folder": os.path.join(MM_MERGEBENCH_ROOT, "Unseen_data", "AOKVQA"),
        "score_kind": "aokvqa",
        "max_new_tokens": 32,
    },
    "imagenet_r": {
        "config_task": "imagenet",
        "task": "imagenet",
        "question_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "ImageNet-R", "test.json"),
        "annotation_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "ImageNet-R", "test.json"),
        "image_folder": os.path.join(MM_MERGEBENCH_ROOT, "Unseen_data"),
        "score_kind": "imagenet_r",
        "max_new_tokens": 32,
    },
    "screen2words": {
        "config_task": "flickr30k",
        "task": "flickr30k",
        "question_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "Screen2words", "test.json"),
        "annotation_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "Screen2words", "test_coco_type.json"),
        "image_folder": os.path.join(MM_MERGEBENCH_ROOT, "Unseen_data"),
        "score_kind": "caption",
        "max_new_tokens": 64,
    },
    "tabmwp": {
        "config_task": "ocr_vqa",
        "task": "ocrvqa",
        "question_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "TabMWP", "test.json"),
        "annotation_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "TabMWP", "test.json"),
        "image_folder": os.path.join(MM_MERGEBENCH_ROOT, "Unseen_data"),
        "score_kind": "tabmwp",
        "max_new_tokens": 32,
    },
}


def canonical_task(task: str) -> str:
    normalized = str(task).strip().lower().replace(" ", "_")
    return TASK_CONFIG_ALIASES.get(normalized, normalized)


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required for LLaVA task configs.")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one refined LLaVA LoRA adapter with LoRA_LLAVA checkpoint_eval")
    parser.add_argument("--task", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--llava-project-dir", required=True)
    parser.add_argument("--llava-config-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--model-base", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-flash-attn", action="store_true")
    return parser.parse_args()


def select_questions(questions: List[Dict[str, Any]], num_chunks: int, max_samples: int | None) -> List[Dict[str, Any]]:
    # The training-time evaluator is single-process. Keep num_chunks accepted for
    # command compatibility, but evaluate the full selected set in this runner.
    selected = list(questions)
    if max_samples is not None:
        selected = selected[: max(0, int(max_samples))]
    return selected


def image_field_aliases(image_file: Any) -> List[str]:
    aliases: List[str] = []
    seen = set()

    def add(candidate: Any) -> None:
        text = str(candidate or "").strip()
        if text and text not in seen:
            seen.add(text)
            aliases.append(text)

    raw = str(image_file or "").strip()
    add(raw)
    normalized = raw.replace("\\", "/")
    if "/" in normalized:
        prefix, filename = normalized.rsplit("/", 1)
        prefix = prefix + "/"
    else:
        prefix, filename = "", normalized
    stem, ext = os.path.splitext(filename)
    if stem:
        if stem.lower().startswith("test_") and len(stem) > 5:
            add(prefix + stem[5:] + ext)
        else:
            add(prefix + "test_" + filename)
    return aliases


def image_root_candidates(image_folder: str, data_path: str = "") -> List[Path]:
    roots: List[Path] = []
    seen = set()

    def add_root(root: Optional[Path]) -> None:
        if root is None:
            return
        try:
            resolved = root.resolve()
        except FileNotFoundError:
            resolved = root
        key = str(resolved)
        if key and key not in seen:
            seen.add(key)
            roots.append(resolved)

    if image_folder:
        image_root = Path(str(image_folder))
        add_root(image_root)
        if image_root.name:
            add_root(image_root.with_name(image_root.name.lower()))
            add_root(image_root.with_name(image_root.name.upper()))
            add_root(image_root.with_name(image_root.name.capitalize()))
        add_root(image_root / "process")
        for parent in image_root.parents:
            add_root(parent)
    if data_path:
        data_root = Path(str(data_path)).parent
        add_root(data_root)
        for parent in data_root.parents:
            add_root(parent)
    return roots


def resolve_image_path_alias(image_file: Any, image_folder: str, data_path: str = "") -> Optional[str]:
    aliases = image_field_aliases(image_file)
    for alias in aliases:
        alias_path = Path(alias)
        if alias_path.is_absolute() and alias_path.is_file():
            return str(alias_path)
    for root in image_root_candidates(image_folder, data_path):
        for alias in aliases:
            candidate = root / alias
            if candidate.is_file():
                return str(candidate)
    return None


def prepare_question_image_paths(
    questions: Sequence[Dict[str, Any]],
    image_folder: str,
    data_path: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    prepared: List[Dict[str, Any]] = []
    patched = 0
    unresolved = 0
    for item in questions:
        row = dict(item)
        image_file = row.get("image")
        if image_file:
            resolved = resolve_image_path_alias(image_file, image_folder, data_path)
            if resolved:
                if str(image_file) != resolved:
                    row["image"] = resolved
                    patched += 1
            else:
                unresolved += 1
        prepared.append(row)
    return prepared, patched, unresolved


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_result_text(path: str, metrics: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            handle.write(f"{key}: {value}\n")


def config_model_base(config: Dict[str, Any]) -> str:
    value = (config.get("model") or {}).get("model_name_or_path", "")
    return "" if value is None else str(value)


def task_config_path(config_dir: str, task: str) -> str:
    return os.path.join(os.path.abspath(config_dir), f"{canonical_task(task)}_lora_config.yaml")


def normalize_image_folder_path(path: Any) -> str:
    text = "" if path is None else str(path)
    if not text or os.path.exists(text):
        return text

    image_root = Path(text)
    candidates: List[Path] = []
    if image_root.name:
        candidates.extend(
            [
                image_root.with_name(image_root.name.lower()),
                image_root.with_name(image_root.name.upper()),
                image_root.with_name(image_root.name.capitalize()),
            ]
        )

    replacements = {
        "/Seen_data/Flickr30k": "/Seen_data/flickr30k",
        "\\Seen_data\\Flickr30k": "\\Seen_data\\flickr30k",
    }
    for old, new in replacements.items():
        if old in text:
            candidates.append(Path(text.replace(old, new)))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return key
    return text


def load_task_config(config_dir: str, task: str) -> Tuple[str, Dict[str, Any]]:
    path = task_config_path(config_dir, task)
    return path, load_yaml(path)


def load_eval_config(config_dir: str, task: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    task = canonical_task(task)
    if task in UNSEEN_EVAL_SPECS:
        spec = dict(UNSEEN_EVAL_SPECS[task])
        config_path, config = load_task_config(config_dir, str(spec["config_task"]))
        base_eval_cfg = dict((config.get("checkpoint_eval") or {}))
        for key in ("conv_mode", "temperature", "top_p", "num_beams", "num_workers", "batch_size"):
            if key in base_eval_cfg and key not in spec:
                spec[key] = base_eval_cfg[key]
        spec["image_folder"] = normalize_image_folder_path(spec.get("image_folder", ""))
        spec["config_path"] = config_path
        return task, config, spec

    config_path, config = load_task_config(config_dir, task)
    eval_cfg = dict(config.get("checkpoint_eval") or {})
    if not eval_cfg:
        raise ValueError(f"No checkpoint_eval block found in {config_path}")
    eval_cfg["image_folder"] = normalize_image_folder_path(eval_cfg.get("image_folder", ""))
    eval_cfg["config_path"] = config_path
    return task, config, eval_cfg


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        return [item for item in payload["annotations"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported annotation format: {path}")


def load_json_payload(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def annotation_answer_for_task(task: str, ann: Dict[str, Any]) -> Any:
    if task == "grounding" and "answer_bbox" in ann:
        return ann.get("answer_bbox")
    for key in ("answer", "multiple_choice_answer", "label", "caption", "text"):
        if key in ann:
            return ann.get(key)
    answers = ann.get("answers")
    if isinstance(answers, list):
        return [item.get("answer", item) if isinstance(item, dict) else item for item in answers]
    if "answer_bbox" in ann:
        return ann.get("answer_bbox")
    return ""


def coco_caption_ground_truths(annotation_file: str) -> Optional[Tuple[List[Any], Dict[str, List[str]]]]:
    payload = load_json_payload(annotation_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
        return None
    image_ids = [image.get("id") for image in payload.get("images", []) if isinstance(image, dict) and "id" in image]
    captions: Dict[str, List[str]] = {}
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or "image_id" not in ann:
            continue
        captions.setdefault(str(ann.get("image_id")), []).append(str(ann.get("caption", "")))
    return image_ids, captions


def write_ans_gt_json(task: str, predictions: List[Dict[str, Any]], annotation_file: str, output_dir: str) -> str:
    output_path = os.path.join(output_dir, "ans_gt.json")
    records: List[Dict[str, Any]] = []
    caption_gt = None
    try:
        caption_gt = coco_caption_ground_truths(annotation_file)
    except Exception:
        caption_gt = None

    if caption_gt is not None:
        image_ids, captions = caption_gt
        for idx, pred in enumerate(predictions):
            image_id = pred.get("image_id")
            if image_id is None and idx < len(image_ids):
                image_id = image_ids[idx]
            if image_id is None:
                image_id = idx + 1
            records.append(
                {
                    "question_id": pred.get("question_id"),
                    "image_id": image_id,
                    "pred": pred.get("text", ""),
                    "ground_truth": captions.get(str(image_id), []),
                }
            )
    else:
        annotation_records = load_json_records(annotation_file)
        annotation_by_id = {str(item.get("question_id")): item for item in annotation_records}
        pair_by_order = task == "imagenet_r"
        for idx, pred in enumerate(predictions):
            ann = annotation_records[idx] if pair_by_order and idx < len(annotation_records) else annotation_by_id.get(str(pred.get("question_id")))
            records.append(
                {
                    "question_id": pred.get("question_id"),
                    "pred": pred.get("text", ""),
                    "ground_truth": annotation_answer_for_task(task, ann or {}),
                }
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    return output_path


def subset_coco_caption_annotation_file(annotation_file: str, output_dir: str, num_predictions: int) -> str:
    payload = load_json_payload(annotation_file)
    if not isinstance(payload, dict) or num_predictions <= 0:
        return annotation_file
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        return annotation_file
    if num_predictions >= len(images):
        return annotation_file

    subset_payload = dict(payload)
    subset_images = images[:num_predictions]
    image_ids = {item.get("id") for item in subset_images if isinstance(item, dict)}
    subset_payload["images"] = subset_images
    subset_payload["annotations"] = [
        item
        for item in annotations
        if isinstance(item, dict) and item.get("image_id") in image_ids
    ]
    output_path = os.path.join(output_dir, "annotation_coco_scoring.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(subset_payload, handle, ensure_ascii=True, indent=2)
    return output_path


def normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(r"^(the answer is|answer is|answer:|response:)\s*", "", text)
    text = text.strip(" \t\r\n.?!,:;\"'`*")
    text = re.sub(r"\s+", " ", text)
    return text


def first_choice_token(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = normalize_match_text(text)
    match = re.search(r"\boption\s*([A-Za-z]|\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*[\(\[]?\s*([A-Za-z]|\d+)(?:\s*[\)\]\.:\-]|\s*$|\b)", cleaned)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(?:the\s+)?answer\s*(?:is|:|=|-)?\s*([A-Za-z]|\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*[\(\[]?\s*([A-Za-z]|\d+)(?:\s*[\)\]\.:\-]|\s*$)", text)
    return match.group(1).upper() if match else ""


def choice_map(prompt: Any) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    for line in str(prompt or "").splitlines():
        match = re.match(r"^\s*([A-Za-z]|\d+)\s*[\.\)]\s*(.+?)\s*$", line)
        if match:
            choices[match.group(1).upper()] = normalize_match_text(match.group(2))
    return choices


def option_answer_match(pred_text: Any, ann: Mapping[str, Any]) -> bool:
    gt = str(ann.get("answer", "")).strip()
    pred_norm = normalize_match_text(pred_text)
    gt_norm = normalize_match_text(gt)
    if pred_norm == gt_norm:
        return True
    token = first_choice_token(pred_text)
    labels = {gt.upper()}
    if gt.isdigit():
        idx = int(gt)
        if 0 <= idx < 26:
            labels.add(chr(ord("A") + idx))
    if token and token in labels:
        return True
    choices = choice_map(ann.get("text", ""))
    option_text = choices.get(gt.upper())
    if option_text and (pred_norm == option_text or option_text in pred_norm):
        return True
    if gt.isdigit():
        label = chr(ord("A") + int(gt)) if int(gt) < 26 else ""
        option_text = choices.get(label)
        if option_text and (pred_norm == option_text or option_text in pred_norm):
            return True
    return False


def numbers_equal(pred_text: Any, answer_text: Any) -> bool:
    pred_numbers = re.findall(r"-?\d+(?:\.\d+)?", str(pred_text or "").replace(",", ""))
    answer_numbers = re.findall(r"-?\d+(?:\.\d+)?", str(answer_text or "").replace(",", ""))
    if not pred_numbers or not answer_numbers:
        return False
    try:
        return abs(float(pred_numbers[0]) - float(answer_numbers[0])) <= 1.0e-6
    except Exception:
        return False


def short_answer_match(pred_text: Any, answer_text: Any) -> bool:
    return numbers_equal(pred_text, answer_text) or normalize_match_text(pred_text) == normalize_match_text(answer_text)


_BBOX_RE = re.compile(
    r"[\[\(]\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*"
    r"[\]\)]"
)


def parse_bbox_numbers(value: Any) -> Optional[List[float]]:
    text = str(value or "")
    match = _BBOX_RE.search(text)
    if match:
        try:
            return [float(item) for item in match.groups()]
        except Exception:
            return None
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) < 4:
        return None
    try:
        return [float(item) for item in numbers[:4]]
    except Exception:
        return None


def normalize_xyxy_box(box: Sequence[float]) -> Optional[List[float]]:
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return None
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        return None
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = min(1.0, max(0.0, left))
    top = min(1.0, max(0.0, top))
    right = min(1.0, max(0.0, right))
    bottom = min(1.0, max(0.0, bottom))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def normalize_pixel_xyxy_box(box: Sequence[float]) -> Optional[List[float]]:
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return None
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        return None
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def annotation_size(ann: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        size = [float(x) for x in ann.get("size", [])]
    except Exception:
        return None
    if len(size) < 2 or size[0] <= 0 or size[1] <= 0:
        return None
    return size[0], size[1]


def square_padded_pixel_xyxy_to_unit(box: Sequence[float], size: Optional[Tuple[float, float]]) -> Optional[List[float]]:
    if size is None:
        return None
    raw = normalize_pixel_xyxy_box(box)
    if raw is None:
        return None
    width, height = size
    max_wh = max(width, height)
    x1, y1, x2, y2 = raw
    if width > height:
        pad = (width - height) / 2.0
        y1 += pad
        y2 += pad
    elif height > width:
        pad = (height - width) / 2.0
        x1 += pad
        x2 += pad
    return normalize_xyxy_box([x1 / max_wh, y1 / max_wh, x2 / max_wh, y2 / max_wh])


def add_grounding_candidate(candidates: List[List[float]], box: Optional[Sequence[float]]) -> None:
    if box is None:
        return
    normalized = normalize_xyxy_box(box)
    if normalized is None:
        return
    for existing in candidates:
        if all(abs(a - b) <= 1.0e-6 for a, b in zip(existing, normalized)):
            return
    candidates.append(normalized)


def grounding_box_candidates(box: Sequence[float], ann: Mapping[str, Any]) -> List[List[float]]:
    size = annotation_size(ann)
    candidates: List[List[float]] = []
    try:
        max_abs = max(abs(float(x)) for x in box)
    except Exception:
        return candidates

    if max_abs <= 1.5:
        add_grounding_candidate(candidates, box)
        return candidates

    add_grounding_candidate(candidates, [float(x) / 1000.0 for x in box])
    if size is not None:
        max_wh = max(size)
        add_grounding_candidate(candidates, [float(x) / max_wh for x in box])
        add_grounding_candidate(candidates, square_padded_pixel_xyxy_to_unit(box, size))
    return candidates


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
    bx1, by1, bx2, by2 = [float(x) for x in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def grounding_iou(pred_text: Any, ann: Mapping[str, Any]) -> float:
    pred_box = parse_bbox_numbers(pred_text)
    gt_box = parse_bbox_numbers(ann.get("answer_bbox", ""))
    if pred_box is None or gt_box is None:
        return 0.0
    gt_candidates = grounding_box_candidates(gt_box, ann)
    pred_candidates = grounding_box_candidates(pred_box, ann)
    if not gt_candidates or not pred_candidates:
        return 0.0
    return max(bbox_iou(pred, gt) for pred in pred_candidates for gt in gt_candidates)


def score_grounding_predictions(
    predictions: Sequence[Dict[str, Any]],
    annotation_file: str,
    output_dir: str,
) -> Dict[str, float]:
    annotations = {str(item.get("question_id")): item for item in load_json_records(annotation_file)}
    total = 0
    correct = 0
    false_answers: List[Dict[str, Any]] = []
    for pred in predictions:
        ann = annotations.get(str(pred.get("question_id")))
        if ann is None:
            continue
        total += 1
        iou = grounding_iou(pred.get("text", ""), ann)
        if iou > 0.5:
            correct += 1
        else:
            row = dict(pred)
            row["ground_truth"] = ann.get("answer_bbox", "")
            row["grounding_iou"] = round(iou, 6)
            false_answers.append(row)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "false_answers.json"), "w", encoding="utf-8") as handle:
        json.dump({"items": false_answers[:1000], "num_false": len(false_answers)}, handle, ensure_ascii=False, indent=2)
    return {
        "Accuracy": 0.0 if total == 0 else float(100.0 * correct / total),
        "Samples": float(total),
    }


def score_unseen_predictions(task: str, predictions: List[Dict[str, Any]], annotation_file: str) -> Dict[str, float]:
    annotation_records = load_json_records(annotation_file)
    annotations = {str(item.get("question_id")): item for item in annotation_records}
    total = 0
    correct = 0
    if task == "imagenet_r":
        # ImageNet-R reuses filenames such as art_1.jpg in many class folders.
        paired_records = zip(predictions, annotation_records)
    else:
        paired_records = ((pred, annotations.get(str(pred.get("question_id")))) for pred in predictions)
    for pred, ann in paired_records:
        if ann is None:
            continue
        total += 1
        pred_text = str(pred.get("text", ""))
        answer = ann.get("answer", "")
        if task == "aokvqa":
            ok = option_answer_match(pred_text, ann)
        elif task == "imagenet_r":
            answer_text = str(answer).strip()
            pred_text = pred_text.strip()
            ok = bool(answer_text and pred_text) and (
                answer_text.upper() in pred_text.upper() or pred_text.upper() in answer_text.upper()
            )
        elif task == "tabmwp":
            ok = short_answer_match(pred_text, answer)
        else:
            raise ValueError(f"No unseen scorer registered for task={task}")
        correct += int(ok)
    return {
        "Accuracy": 0.0 if total == 0 else float(100.0 * correct / total),
        "Samples": float(total),
    }


def main() -> None:
    args = parse_args()
    llava_project_dir = os.path.abspath(os.path.expanduser(args.llava_project_dir))
    if llava_project_dir not in sys.path:
        sys.path.insert(0, llava_project_dir)

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.train.checkpoint_eval import SeenTaskCheckpointEvaluator
    from llava.utils import disable_torch_init

    task, config, eval_cfg = load_eval_config(args.llava_config_dir, args.task)

    model_base = args.model_base or config_model_base(config)
    if not model_base:
        raise ValueError("model_base is empty. Set --model-base or model.model_name_or_path in the LLaVA config.")

    eval_task_name = str(eval_cfg.get("task") or task)
    os.makedirs(args.output_dir, exist_ok=True)

    disable_torch_init()
    model_name = get_model_name_from_path(args.adapter_dir)
    if "llava" not in model_name.lower() or "lora" not in model_name.lower():
        model_name = "llava_lora_meta_ebm"
    tokenizer, model, _, _ = load_pretrained_model(
        os.path.abspath(args.adapter_dir),
        model_base,
        model_name,
        device=args.device,
        use_flash_attn=bool(args.use_flash_attn),
    )
    model.eval()

    evaluator = SeenTaskCheckpointEvaluator(
        task_name=eval_task_name,
        question_file=str(eval_cfg.get("question_file", "")),
        annotation_file=eval_cfg.get("annotation_file"),
        base_dir=eval_cfg.get("base_dir"),
        image_folder=str(eval_cfg.get("image_folder", "")),
        output_dir=args.output_dir,
        conv_mode=str(eval_cfg.get("conv_mode", "vicuna_v1")),
        temperature=float(eval_cfg.get("temperature", 0.0)),
        top_p=eval_cfg.get("top_p"),
        num_beams=int(eval_cfg.get("num_beams", 1)),
        max_new_tokens=int(eval_cfg.get("max_new_tokens", 128)),
        num_workers=int(eval_cfg.get("num_workers", 0)),
        batch_size=int(eval_cfg.get("batch_size", 1)),
    )

    questions = select_questions(evaluator._load_questions(), args.num_chunks, args.max_samples)
    questions, patched_images, unresolved_images = prepare_question_image_paths(
        questions,
        str(eval_cfg.get("image_folder", "")),
        str(eval_cfg.get("question_file", "")),
    )
    if patched_images:
        print(f"[info] resolved {patched_images} image paths with Qwen-style aliases")
    if unresolved_images:
        print(f"[warn] {unresolved_images} image paths were not pre-resolved; falling back to LoRA_LLAVA resolver")
    with torch.inference_mode():
        predictions = evaluator._generate_predictions(model, tokenizer, questions)
    write_jsonl(args.predictions_file, predictions)
    annotation_file = str(eval_cfg.get("annotation_file") or eval_cfg.get("question_file") or "")
    if annotation_file:
        write_ans_gt_json(task, predictions, annotation_file, args.output_dir)
    score_kind = str(eval_cfg.get("score_kind", "") or "")
    if score_kind in {"aokvqa", "imagenet_r", "tabmwp"}:
        metrics = score_unseen_predictions(task, predictions, str(eval_cfg["annotation_file"]))
    elif evaluator.task_name == "grounding":
        metrics = score_grounding_predictions(predictions, str(eval_cfg["annotation_file"]), args.output_dir)
    elif evaluator.task_name in {"flickr30k", "vizwiz"}:
        original_annotation_file = evaluator.annotation_file
        if original_annotation_file:
            evaluator.annotation_file = subset_coco_caption_annotation_file(
                str(original_annotation_file),
                args.output_dir,
                len(predictions),
            )
        metrics = evaluator._evaluate_predictions(predictions, args.output_dir)
        evaluator.annotation_file = original_annotation_file
    else:
        metrics = evaluator._evaluate_predictions(predictions, args.output_dir)

    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
    with open(args.metrics_file, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    write_result_text(os.path.join(args.output_dir, "Result.text"), metrics)


if __name__ == "__main__":
    main()
