# -----------------------------------------------------------
# Import Libraries and Modules
# -----------------------------------------------------------
import json                           # For JSON file handling
import os                             # For operating system file path operations
import re                             # For regular expressions (if needed)
from typing import Tuple, List, Optional  # For type annotations
from PIL import Image                 # For image processing

# Import necessary components from Hugging Face Transformers and PEFT
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,  # For quantization (currently not used)
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig  # For parameter-efficient fine-tuning using LoRA
import torch                          # For PyTorch operations

# -----------------------------------------------------------
# Define Tasks and Load Dataset
# -----------------------------------------------------------
# List of tasks for cumulative curriculum training
tasks = ['cathno_det', 'catheter_det', 'artery_cls', 'cadica_det', 'health_cls', 'severity_cls', 'syntax_det']

# Load the dataset from a JSON file
train_path = "complete.json"
with open(train_path) as f:
    train_ds = json.load(f)

# Set up randomness for reproducibility
from random import shuffle
import random
random.seed(42)

# -----------------------------------------------------------
# Model and Processor Setup
# -----------------------------------------------------------
# Define the model ID (you can change it to your preferred model)
model_id = "google/paligemma2-3b-pt-224"  # or your favorite PaliGemma

# Initialize the processor for handling both text and image inputs
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Specify device for computation (GPU)
device = "cuda"

# Load the pre-trained model with custom attention implementation and device mapping.
# Optionally, you can enable quantization by configuring BitsAndBytesConfig.
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    attn_implementation='eager'
    # , quantization_config=bnb_config  # Uncomment to enable 4-bit quantization
)

# Save the model's data type for later use in tensor conversion
DTYPE = model.dtype

# Optionally, if starting from a pre-trained model, you might freeze the base model and only train the head:
# for name, param in model.named_parameters():
#     if "lora_A" in name or "lora_B" in name:
#         param.requires_grad = True

# -----------------------------------------------------------
# LoRA Configuration for Efficient Fine-Tuning
# -----------------------------------------------------------
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices (controls the size of the additional trainable parameters)
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",  # Specify the task type (causal language modeling)
)

# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters summary for verification
model.print_trainable_parameters()
# Example output:
# trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344

# -----------------------------------------------------------
# Cumulative Curriculum Training Loop
# -----------------------------------------------------------
# Initialize a list to keep track of past tasks for curriculum learning
past_tasks = []

# Loop over each task defined in 'tasks'
for task in tasks:
    
    # Initialize training data list for the current task, including a 10% sample from previous tasks
    train_data = []
    
    # For each previously seen task, sample 10% of its data and add to the training data
    for past_task in past_tasks:
        sample_size = int(0.1 * len(train_ds["train"][past_task]))
        for i in random.sample(train_ds["train"][past_task], sample_size):
            train_data.append({
                "image": os.path.join("/home/maxim.popov/paligemma/thesis/paligemma_dataset/train", past_task, i["image"]),
                "prefix": i["prefix"],
                "suffix": i["suffix"]
            })
    
    # Add all instances of the current task to the training data
    for i in train_ds["train"][task]:
        train_data.append({
            "image": os.path.join("/home/maxim.popov/paligemma/thesis/paligemma_dataset/train", task, i["image"]),
            "prefix": i["prefix"],
            "suffix": i["suffix"]
        })
    
    # -----------------------------------------------------------
    # Training Arguments Configuration
    # -----------------------------------------------------------
    args = TrainingArguments(
        num_train_epochs=5,                   # Number of training epochs
        remove_unused_columns=False,          # Do not remove any columns from the dataset
        per_device_train_batch_size=1,        # Batch size per device (GPU)
        gradient_accumulation_steps=4,        # Accumulate gradients over multiple steps to simulate a larger batch size
        warmup_steps=2,                       # Warmup steps for the learning rate scheduler
        learning_rate=2e-5,                   # Learning rate for the optimizer
        weight_decay=1e-6,                    # Weight decay for regularization
        adam_beta2=0.999,                     # Second beta parameter for the Adam optimizer
        logging_steps=100,                    # Log training metrics every 100 steps
        optim="adamw_torch",                  # Optimizer type (alternatively, use paged optimizers for QLoRA)
        save_steps=10,                        # Save a checkpoint every 10 steps
        save_total_limit=1,                   # Limit the total number of saved checkpoints to 1
        output_dir=f"/l/users/maxim.popov/cardiogpt/paligemma_textqa_random10percent_cumulative_curriculum_5ep",
                                              # Output directory for model checkpoints
        bf16=True,                            # Use bfloat16 precision if supported by your hardware
        report_to=["tensorboard"],            # Report training metrics to TensorBoard
        dataloader_pin_memory=False,          # Whether to pin memory in the DataLoader (set True if needed)
        # max_steps=2000,                     # Optionally, set a maximum number of training steps
        save_strategy="best",                 # Save only the best model based on evaluation metric
        metric_for_best_model="loss"          # Metric used to determine the best model (lower loss is better)
        # Uncomment the following lines to enable evaluation during training:
        # do_eval=True,
        # eval_strategy="epoch"
    )
    
    # -----------------------------------------------------------
    # Define Custom Data Collation Function
    # -----------------------------------------------------------
    def collate_fn(examples):
        # Prepend a special token to each prefix text
        texts = ["<image>" + x["prefix"] for x in examples]
        # Open each image and convert to RGB using PIL
        images = [Image.open(x["image"]).convert("RGB") for x in examples]
        # Extract suffix labels from the examples
        labels = [x["suffix"] for x in examples]
        # Process texts and images using the processor to generate token tensors.
        # Padding and truncation are applied to maintain a consistent sequence length of 128 tokens.
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(DTYPE).to("cuda")
        return tokens
    
    # -----------------------------------------------------------
    # Initialize Trainer and Train Model
    # -----------------------------------------------------------
    trainer = Trainer(
        model=model,               # The model to fine-tune
        train_dataset=train_data,  # Training dataset (current task data + sampled data from past tasks)
        # eval_dataset=eval_ds,    # Optionally, add evaluation dataset
        data_collator=collate_fn,  # Custom collate function for data processing
        args=args                # Training arguments defined above
    )
    
    # Train the model on the current training data
    trainer.train()
    
    # After training, push the fine-tuned model to the Hugging Face Hub
    trainer.push_to_hub()
    
    # Add the current task to past_tasks for cumulative curriculum learning
    past_tasks.append(task)