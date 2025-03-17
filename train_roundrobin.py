# =============================================================================
# Imports and Global Configuration
# =============================================================================
import json
import os
import random
import torch
import numpy as np
import re
from PIL import Image
import supervision as sv
from tqdm import tqdm  # For progress tracking

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,  # For quantization (currently not used)
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig  # For parameter-efficient fine-tuning using LoRA

# Set random seed for reproducibility
random.seed(42)

# Define global variables
tasks = ['cathno_det', 'catheter_det', 'artery_cls', 'cadica_det', 'health_cls', 'severity_cls', 'syntax_det']
CLASSES = ["catheter", "stenosis"]
train_path = "complete.json"

# =============================================================================
# Data Loading
# =============================================================================
with open(train_path) as f:
    train_ds = json.load(f)

# =============================================================================
# Model and Processor Setup
# =============================================================================
model_id = "google/paligemma2-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)
device = "cuda"

# Load the pre-trained model with custom attention implementation and device mapping
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    device_map=device,
    attn_implementation='eager'
)
DTYPE = model.dtype  # Save the model's data type for tensor conversions

# =============================================================================
# LoRA Configuration for Efficient Fine-Tuning
# =============================================================================
# Optionally, if starting from a pre-trained model, freeze base layers (commented out)
# for name, param in model.named_parameters():
#     if "lora_A" in name or "lora_B" in name:
#         param.requires_grad = True

lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices (controls the size of added trainable parameters)
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",  # Specify the task type (causal language modeling)
)
# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Verify trainable parameter summary

# =============================================================================
# Helper Functions
# =============================================================================
def collate_fn(examples):
    """
    Prepare a batch of samples for training.
    Converts texts and images into model-ready tensors.
    """
    texts = ["<image>" + x["prefix"] for x in examples]
    images = [Image.open(x["image"]).convert("RGB") for x in examples]
    labels = [x["suffix"] for x in examples]
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(DTYPE).to(device)
    return tokens


def from_pali_gemma(response: str, resolution_wh):
    """
    Parse the model response using a regex pattern to extract bounding boxes.
    Returns a supervision Detections object with adjusted coordinates.
    """
    _SEGMENT_DETECT_RE = re.compile(
        r'(.*?)' + r'<loc(\d{4})>' * 4 +
        r'\s*(?:%s)?' % (r'<seg(\d{3})>' * 16) +
        r'\s*([^;<>]+)? ?(?:; )?'
    )
    width, height = resolution_wh
    xyxy_list = []
    while response:
        m = _SEGMENT_DETECT_RE.match(response)
        if not m:
            break
        gs = list(m.groups())
        before = gs.pop(0)
        # Convert normalized coordinates to actual pixel values
        y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
        y1, x1, y2, x2 = map(round, (y1 * height, x1 * width, y2 * height, x2 * width))
        content = m.group()
        if before:
            response = response[len(before):]
            content = content[len(before):]
        xyxy_list.append([x1, y1, x2, y2])
        response = response[len(content):]
    xyxy = np.array(xyxy_list)
    return sv.Detections(xyxy=xyxy, class_id=np.array([0]), data={'class_name': [""]})


def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    bbox2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def sample_is_incorrect(sample, task_name):
    """
    Determine whether a sample is processed incorrectly by the model.
    For detection tasks, compare predicted and ground-truth bounding boxes;
    for other tasks, compare generated response with the expected suffix.
    """
    image = Image.open(sample["image"]).convert("RGB")
    prompt = "<image>" + sample["prefix"]
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device).to(torch.half)
    output = model.generate(**inputs, max_new_tokens=256, use_cache=False)
    response = processor.decode(output[0], skip_special_tokens=True).split("\n")[-1]

    if "_det" in task_name:
        try:
            pred_box = from_pali_gemma(response, image.size).xyxy
            gt_box = from_pali_gemma(sample["suffix"], image.size).xyxy
            if len(pred_box) == 0 or len(gt_box) == 0:
                return True
            return iou(pred_box[0], gt_box[0]) < 0.5
        except:
            return True
    else:
        return response.lower().strip() != sample["suffix"].lower().strip()


# =============================================================================
# Main Training Loop (Continual Curriculum Learning)
# =============================================================================
past_tasks_data = {}  # To store samples for past tasks
past_tasks = []       # To keep track of completed tasks

for epoch in range(25):
    for task in tasks:
        print(f"Starting task: {task}")
        train_data = []
        
        # Limit the size of the past_tasks list (if necessary)
        if len(past_tasks) == 7:
            past_tasks.pop(0)
        
        # -----------------------------
        # Priority Replay from Past Tasks
        # -----------------------------
        print("Generating replay queue")
        for past_task in tqdm(past_tasks, desc="Replaying past tasks"):
            past_data_samples = train_ds["train"][past_task]
            errors = []
            random.shuffle(past_data_samples)

            # Collect samples with incorrect predictions
            for sample in past_data_samples:
                sample["image"] = os.path.join(
                    "/home/maxim.popov/paligemma/thesis/paligemma_dataset/train",
                    past_task,
                    sample["image"]
                )
                if sample_is_incorrect(sample, past_task):
                    errors.append(sample)
                if len(errors) >= int(0.1 * len(past_data_samples)):
                    break

            # If insufficient errors, add additional samples from non-error instances
            if len(errors) < int(0.1 * len(past_data_samples)):
                remaining_needed = int(0.1 * len(past_data_samples)) - len(errors)
                non_errors = [s for s in past_data_samples if s not in errors]
                errors += random.sample(non_errors, min(remaining_needed, len(non_errors)))

            train_data.extend(errors)

        # -----------------------------
        # Add Current Task Samples
        # -----------------------------
        current_task_samples = []
        for i in train_ds["train"][task]:
            sample = {
                "image": os.path.join("/home/maxim.popov/paligemma/thesis/paligemma_dataset/train", task, i["image"]),
                "prefix": i["prefix"],
                "suffix": i["suffix"]
            }
            train_data.append(sample)
            current_task_samples.append(sample)

        # Store current task samples for future replay
        past_tasks_data[task] = current_task_samples

        # =============================================================================
        # Training Arguments for Current Epoch
        # =============================================================================
        args = TrainingArguments(
            num_train_epochs=1,
            remove_unused_columns=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_torch",
            save_steps=10,
            save_total_limit=1,
            output_dir=f"/l/users/maxim.popov/cardiogpt/paligemma_textqa_continual_roundrobin",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False,
            save_strategy="best",
            metric_for_best_model="loss"
        )

        # =============================================================================
        # Initialize Trainer and Train on Current Data
        # =============================================================================
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            data_collator=collate_fn,
            args=args
        )

        trainer.train()
        trainer.push_to_hub()

        # Append the current task to the list of past tasks for future replay
        past_tasks.append(task)