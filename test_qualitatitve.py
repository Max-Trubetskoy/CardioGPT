# =============================================================================
# Imports and Global Setup
# =============================================================================
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
import torch
import json
import re
import numpy as np
import supervision as sv
from typing import Tuple, List, Optional
from PIL import Image
from collections import defaultdict
from sklearn.metrics import confusion_matrix

# Set random seed for reproducibility
import random
random.seed(42)


# =============================================================================
# Helper Functions
# =============================================================================
def from_pali_gemma(
    response: str,
    resolution_wh: Tuple[int, int],
    class_list: Optional[List[str]] = None
) -> sv.Detections:
    """
    Parse the model's response to extract bounding box coordinates.

    The response is expected to contain segments with location tags.
    Coordinates are normalized (divided by 1024) and then scaled to the image resolution.
    
    Args:
        response (str): The raw response string from the model.
        resolution_wh (Tuple[int, int]): The width and height of the image.
        class_list (Optional[List[str]]): Optional list of class names.
    
    Returns:
        sv.Detections: A detections object with extracted bounding boxes.
    """
    _SEGMENT_DETECT_RE = re.compile(
        r'(.*?)' +
        r'<loc(\d{4})>' * 4 + r'\s*' +
        '(?:%s)?' % (r'<seg(\d{3})>' * 16) +
        r'\s*([^;<>]+)? ?(?:; )?'
    )
    width, height = resolution_wh
    xyxy_list = []
    class_name_list = []

    while response:
        m = _SEGMENT_DETECT_RE.match(response)
        if not m:
            break

        gs = list(m.groups())
        before = gs.pop(0)
        # Extract normalized coordinates and scale to image size
        y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
        y1, x1, y2, x2 = map(round, (y1 * height, x1 * width, y2 * height, x2 * width))
        content = m.group()
        if before:
            response = response[len(before):]
            content = content[len(before):]

        xyxy_list.append([x1, y1, x2, y2])
        # Optionally, process class name here if needed:
        # class_name_list.append(name.strip())
        response = response[len(content):]

    xyxy = np.array(xyxy_list)

    # If class_list is provided, mapping can be performed here.
    if class_list is None:
        class_id = None
    else:
        # Placeholder for class id mapping
        pass

    return sv.Detections(
        xyxy=xyxy,
        class_id=np.array([0]),  # Currently hard-coded
        data={'class_name': [""]}
    )


def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes in xyxy format.

    Args:
        bbox1: Bounding box 1 [x_min, y_min, x_max, y_max]
        bbox2: Bounding box 2 [x_min, y_min, x_max, y_max]
    
    Returns:
        float: IoU value.
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2

    # Calculate intersection
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    bbox2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# =============================================================================
# Data Loading and Model Setup
# =============================================================================
# Load dataset from JSON
train_path = "complete.json"
with open(train_path) as f:
    train_ds = json.load(f)

# Set model and processor IDs
model_id = "google/paligemma2-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Load the model with half precision and move to GPU
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.half
).to("cuda")

# =============================================================================
# Evaluation: Health Classification
# =============================================================================
# Initialize counters for health classification evaluation
tp, fp, fn, tn, ood = 0, 0, 0, 0, 0
prompt = "<image>In Is this vessel healthy or diseased?"
print(prompt)

# Evaluate on test data for health classification
for instance in train_ds["test"]["health_cls"]:
    image_path = "/home/maxim.popov/paligemma/thesis/paligemma_dataset/test/health_cls/" + instance["image"]
    correct = instance["suffix"].lower()
    raw_image = Image.open(image_path)
    
    inputs = processor(
        text=prompt,
        images=raw_image.convert("RGB"),
        return_tensors="pt"
    ).to('cuda').to(torch.half)
    
    output = model.generate(**inputs, max_new_tokens=20, use_cache=False)
    response = processor.decode(output[0], skip_special_tokens=True).lower().split("\n")[-1]
    
    if response == "healthy" and response == correct:
        tn += 1
    elif response == "diseased" and response == correct:
        tp += 1
    elif response == "healthy" and response != correct:
        fn += 1
    elif response == "diseased" and response != correct:
        fp += 1
    else:
        ood += 1

print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"TN: {tn}")
print(f"FN: {fn}")
print(f"OOD: {ood}")
print(f"Precision: {tp/(tp+fp)}")
print(f"Recall: {tp/(tp+fn)}")
print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn)}")


# =============================================================================
# Evaluation: Artery Classification
# =============================================================================
y_true = []
y_pred = []
prompt = "<image>What artery is in the focus here (RCA/LAD/LCX)?"
print(prompt)

for instance in train_ds["test"]["artery_cls"]:
    image_path = "/home/maxim.popov/paligemma/thesis/paligemma_dataset/test/artery_cls/" + instance["image"]
    correct = instance["suffix"].lower()
    raw_image = Image.open(image_path)
    
    inputs = processor(
        text=prompt,
        images=raw_image.convert("RGB"),
        return_tensors="pt"
    ).to('cuda').to(torch.half)
    
    output = model.generate(**inputs, max_new_tokens=20, use_cache=False)
    response = processor.decode(output[0], skip_special_tokens=True).lower().split("\n")[-1]
    
    if "right" in response or "rca" in response:
        y_pred.append(0)
    elif "left anterior descending" in response or "lad" in response:
        y_pred.append(1)
    elif "left circumflex" in response or "lcx" in response:
        y_pred.append(2)
    else:
        y_pred.append(3)

    if "right coronary artery" in correct:
        y_true.append(0)
    elif "left anterior descending" in correct:
        y_true.append(1)
    elif "left circumflex" in correct:
        y_true.append(2)
    else:
        y_true.append(3)

conf = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf)
print("Accuracy:", (conf * conf.shape[0]).sum() / conf.sum())


# =============================================================================
# Evaluation: Severity Classification (Confusion Matrix)
# =============================================================================
y_true = []
y_pred = []
prompt = "<image> What is the severity of the disease (low/mild/significant/severe/critical/total occlusion)?"
print(prompt)

for instance in train_ds["test"]["severity_cls"]:
    image_path = "/home/maxim.popov/paligemma/thesis/paligemma_dataset/test/severity_cls/" + instance["image"]
    correct = instance["suffix"].lower()
    raw_image = Image.open(image_path)
    
    inputs = processor(
        text=prompt,
        images=raw_image.convert("RGB"),
        return_tensors="pt"
    ).to('cuda').to(torch.half)
    
    output = model.generate(**inputs, max_new_tokens=20, use_cache=False)
    response = processor.decode(output[0], skip_special_tokens=True).lower().split("\n")[-1].lower()
    
    if "low" in response:
        y_pred.append(0)
    elif "mild" in response:
        y_pred.append(1)
    elif "significant" in response:
        y_pred.append(2)
    elif "severe" in response:
        y_pred.append(3)
    elif "critical" in response:
        y_pred.append(4)
    elif "near total occlusion" in response:
        y_pred.append(5)
    elif "total occlusion" in response:
        y_pred.append(6)
    else:
        y_pred.append(7)

    if "low" in correct:
        y_true.append(0)
    elif "mild" in correct:
        y_true.append(1)
    elif "significant" in correct:
        y_true.append(2)
    elif "severe" in correct:
        y_true.append(3)
    elif "critical" in correct:
        y_true.append(4)
    elif "near total occlusion" in correct:
        y_true.append(5)
    elif "total occlusion" in correct:
        y_true.append(6)
    else:
        y_true.append(7)

conf = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf)
print("Accuracy:", (conf * conf.shape[0]).sum() / conf.sum())



# =============================================================================
# Evaluation: Catheter Detection
# =============================================================================
prompt = "<image>detect catheter"
print(prompt)
metrics = {}
split = "test"
task = "catheter_det"

with torch.amp.autocast("cuda"):
    with torch.no_grad():
        for i in train_ds[split][task]:
            try:
                raw_image = Image.open(f"/home/maxim.popov/paligemma/thesis/paligemma_dataset/{split}/{task}/" + i["image"])
                inputs = processor(
                    text=prompt,
                    images=raw_image.convert("RGB"),
                    return_tensors="pt"
                ).to('cuda').to(torch.half)
                output = model.generate(**inputs, max_new_tokens=20, use_cache=False)
                # Extract classes from prefix if present
                CLASSES = i.get('prefix').replace("detect ", "").split(" ; ")
                response = processor.decode(output[0], skip_special_tokens=True)
                # Get detections and ground truth bounding boxes
                detections = from_pali_gemma(response.split("\n")[-1], raw_image.size, ["catheter", "stenosis"]).xyxy[0]
                gt = from_pali_gemma(i["suffix"], raw_image.size, CLASSES).xyxy[0]
                metrics[i["image"]] = iou(detections, gt)
            except:
                metrics[i["image"]] = 0.0
                continue

print("Catheter AUC:", torch.Tensor(list(metrics.values())).mean(), "+-", torch.Tensor(list(metrics.values())).std())


# =============================================================================
# Evaluation: Cadica Detection (Stenosis)
# =============================================================================
metrics = defaultdict(list)
split = "test"
task = "cadica_det"
with torch.amp.autocast("cuda"):
    with torch.no_grad():
        for i in train_ds[split][task]:
            try:
                raw_image = Image.open(f"/home/maxim.popov/paligemma/thesis/paligemma_dataset/{split}/{task}/" + i["image"])
                prompt = "<image>" + i.get("prefix").lower()
                inputs = processor(
                    text=prompt,
                    images=raw_image.convert("RGB"),
                    return_tensors="pt"
                ).to('cuda').to(torch.half)
                output = model.generate(**inputs, max_new_tokens=20, use_cache=False)
                CLASSES = prompt.replace("<image>detect ", "").split(" ; ")
                response = processor.decode(output[0], skip_special_tokens=True).split("\n")[-1].lower()
                
                detections_metrics = []
                for true_instance in i["suffix"].split(";"):
                    gt = from_pali_gemma(true_instance, raw_image.size, CLASSES).xyxy[0]
                    instance_metrics = []
                    for instance in response.split(";"):
                        if "<loc" in instance and len(instance) > 36:
                            detections = from_pali_gemma(instance, raw_image.size, CLASSES).xyxy[0]
                            instance_metrics.append(iou(detections, gt))
                    instance_metrics = torch.Tensor(instance_metrics).max()
                    detections_metrics.append(instance_metrics)
                metrics[i["image"]] = torch.Tensor(detections_metrics).mean()

            except:
                metrics[i["image"]] = 0.0
                continue

print("Stenosis AUC:", torch.Tensor(list(metrics.values())).mean(), "+-", torch.Tensor(list(metrics.values())).std())


# =============================================================================
# Evaluation: Syntax Detection
# =============================================================================
metrics = defaultdict(list)
class_metrics = defaultdict(list)
split = "test"
task = "syntax_det"
with torch.amp.autocast("cuda"):
    with torch.no_grad():
        for i in train_ds[split][task]:
            try:
                raw_image = Image.open(f"/home/maxim.popov/paligemma/thesis/paligemma_dataset/{split}/{task}/" + i["image"])
                prompt = "<image>" + i.get("prefix").lower()
                inputs = processor(
                    text=prompt,
                    images=raw_image.convert("RGB"),
                    return_tensors="pt"
                ).to('cuda').to(torch.half)
                output = model.generate(**inputs, max_new_tokens=20, use_cache=False)
                CLASSES = prompt.replace("<image>detect ", "").split(" ; ")
                response = processor.decode(output[0], skip_special_tokens=True).split("\n")[-1].lower()

                detections_metrics = []
                for true_instance in i["suffix"].split(";"):
                    gt = from_pali_gemma(true_instance, raw_image.size, CLASSES).xyxy[0]
                    instance_metrics = []
                    for instance in response.split(";"):
                        if "<loc" in instance and len(instance) > 36:
                            detections = from_pali_gemma(instance, raw_image.size, CLASSES).xyxy[0]
                            instance_metrics.append(iou(detections, gt))
                    instance_metrics = torch.Tensor(instance_metrics).max()
                    detections_metrics.append(instance_metrics)
                metric = torch.Tensor(detections_metrics).mean()
                class_metrics[CLASSES[0]].append(metric)

            except:
                print(i["image"], "caused error")
                metrics[i["image"]] = 0.0
                continue

print("Syntax AUC:", torch.Tensor(list(metrics.values())).mean(), "+-", torch.Tensor(list(metrics.values())).std())
print("Syntax AUC per class:")
for k, v in class_metrics.items():
    print(k, torch.Tensor(v).mean(), "+-", torch.Tensor(v).std())
