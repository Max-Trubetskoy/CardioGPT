# -----------------------------------------------------------
# Import Libraries and Modules
# -----------------------------------------------------------
import json                          # For JSON file operations
import os                            # For operating system interactions
import random                        # For random sampling and shuffling
import torch                         # For PyTorch operations
import numpy as np                   # For numerical operations
import re                            # For regular expressions
from PIL import Image                # For image processing
import supervision as sv             # Custom/external supervision module

# Import components from Hugging Face Transformers and PEFT
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,              # For quantization (currently not used)
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig  # For parameter-efficient fine-tuning using LoRA

# Set a fixed random seed for reproducibility
random.seed(42)

# -----------------------------------------------------------
# Define Tasks and Load Dataset
# -----------------------------------------------------------
# List of tasks for training
tasks = ['cathno_det', 'catheter_det', 'artery_cls', 'cadica_det', 'health_cls', 'severity_cls', 'syntax_det']
# Define classes (if needed elsewhere)
CLASSES = ["catheter", "stenosis"]

# Load training dataset from a JSON file
train_path = "complete.json"
with open(train_path) as f:
    train_ds = json.load(f)

# -----------------------------------------------------------
# Model and Processor Setup
# -----------------------------------------------------------
model_id = "google/paligemma2-3b-pt-224"  # Specify your model ID
processor = PaliGemmaProcessor.from_pretrained(model_id)  # Initialize the processor
device = "cuda"  # Use GPU for training

# Load the pre-trained model with custom attention implementation and device mapping
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    device_map=device,
    attn_implementation='eager'
)

# Save the model's data type for future tensor conversions
DTYPE = model.dtype

# Optionally, if starting from a pre-trained model, freeze base layers (commented out)
# for name, param in model.named_parameters():
#     if "lora_A" in name or "lora_B" in name:
#         param.requires_grad = True

# -----------------------------------------------------------
# LoRA Configuration for Efficient Fine-Tuning
# -----------------------------------------------------------
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices (controls the size of added trainable parameters)
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",  # Specify the task type (causal language modeling)
)

# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Print a summary of trainable parameters for verification
model.print_trainable_parameters()
# Example output:
# trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344

# -----------------------------------------------------------
# Define Helper Functions
# -----------------------------------------------------------
def collate_fn(examples):
    """
    Custom collate function to process a batch of examples.
    Processes text, image, and suffix into model-ready tensors.
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
    Parses the model response using a regex pattern and returns detections.
    Converts detected coordinates to the image's resolution.
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
    Computes the Intersection over Union (IoU) of two bounding boxes.
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
    Determines whether a sample is incorrectly processed by the model.
    For detection tasks, compares predicted and ground-truth bounding boxes.
    For other tasks, compares generated text with the expected suffix.
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

# -----------------------------------------------------------
# Curriculum Training Setup
# -----------------------------------------------------------
past_tasks_data = {}  # To store samples for past tasks
past_tasks = []       # To keep track of completed tasks

# Loop over each task for continual curriculum training
for task in tasks:
    print(f"Starting task: {task}")
    
    # For each task, train for 5 epochs
    for epoch in range(5):
        train_data = []
        
        # Priority replay: retain hardest samples from past tasks
        for past_task in past_tasks:
            past_data_samples = past_tasks_data[past_task]
            errors = []
            random.shuffle(past_data_samples)
            
            # Collect a sample of incorrect predictions (at least 10% of past task data)
            for sample in past_data_samples:
                if sample_is_incorrect(sample, past_task):
                    errors.append(sample)
                if len(errors) >= int(0.1 * len(past_data_samples)):
                    break
            
            # If insufficient errors, sample additional non-error instances
            if len(errors) < int(0.1 * len(past_data_samples)):
                remaining_needed = int(0.1 * len(past_data_samples)) - len(errors)
                non_errors = [s for s in past_data_samples if s not in errors]
                errors += random.sample(non_errors, min(remaining_needed, len(non_errors)))
            
            train_data.extend(errors)
        
        # Add all current task samples to the training data
        current_task_samples = []
        for i in train_ds["train"][task]:
            sample = {
                "image": os.path.join("/home/maxim.popov/paligemma/thesis/paligemma_dataset/train", task, i["image"]),
                "prefix": i["prefix"],
                "suffix": i["suffix"]
            }
            train_data.append(sample)
            current_task_samples.append(sample)
        
        # Store current task samples for future curriculum replay
        past_tasks_data[task] = current_task_samples
        
        # -----------------------------------------------------------
        # Training Arguments Configuration for Current Epoch
        # -----------------------------------------------------------
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
            output_dir=f"/l/users/maxim.popov/cardiogpt/paligemma_textqa_continual_priority2_{task}_ep{epoch}",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False,
            save_strategy="best",
            metric_for_best_model="loss"
        )
        
        # -----------------------------------------------------------
        # Initialize Trainer and Train on Current Epoch Data
        # -----------------------------------------------------------
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            data_collator=collate_fn,
            args=args
        )
        
        trainer.train()
    
    # After all epochs for the current task, push the model to the Hugging Face Hub
    trainer.push_to_hub()
    
    # Add the current task to the list of past tasks for curriculum learning
    past_tasks.append(task)
