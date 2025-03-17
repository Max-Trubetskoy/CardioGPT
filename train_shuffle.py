# -----------------------------------------------------------
# Import Libraries and Modules
# -----------------------------------------------------------
import json                            # For reading and writing JSON files
import os                              # For file and directory operations
from PIL import Image                  # For image processing

# Import necessary components from Hugging Face Transformers and PEFT
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig   # For quantization (currently not used)
from peft import get_peft_model, LoraConfig    # For applying LoRA for efficient fine-tuning
import torch                           # For PyTorch operations
from transformers import TrainingArguments, Trainer  # For configuring and running training

# -----------------------------------------------------------
# Model and Processor Setup
# -----------------------------------------------------------
# Define the model ID (you can replace it with your preferred model)
model_id = "google/paligemma2-3b-pt-224"  # or your favorite PaliGemma model

# Initialize the processor for handling text and image inputs
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Set the device to CUDA for GPU acceleration
device = "cuda"

# Load the pre-trained model with half precision (fp16) for improved efficiency, and move it to the GPU
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.half  # Use half precision to reduce memory usage
).to(device)

# Disable the model's caching mechanism to improve training performance
model.config.use_cache = False

# Optional: Configure 4-bit quantization (currently commented out)
# bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

# -----------------------------------------------------------
# LoRA (Low-Rank Adaptation) Configuration
# -----------------------------------------------------------
# Define LoRA settings for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices (controls the added trainable parameters)
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # Target modules to apply LoRA
    task_type="CAUSAL_LM",  # Task type: Causal Language Modeling
)

# Reload the model with custom attention implementation and device mapping.
# Optionally, include the quantization configuration if needed.
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    attn_implementation='eager'
    # , quantization_config=bnb_config  # Uncomment to enable 4-bit quantization
)

#-----------------------------------------------------------
# Starting from a pre-trained LoRA model
#-----------------------------------------------------------
# if starting from a pre-trained model, you can freeze the base model and only train the head
# for name, param in model.named_parameters():
#     if "lora_A" in name or "lora_B" in name:
#         param.requires_grad = True

#------------------------------------------------------------
# Apply LoRA configuration to the base model
#------------------------------------------------------------

# Apply LoRA configuration to the model for fine-tuning
model = get_peft_model(model, lora_config)

# Print trainable parameters: shows trainable vs. total parameters and percentage
model.print_trainable_parameters()
# Example output:
# trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344

# Save the model's data type for later tensor conversion
DTYPE = model.dtype

# -----------------------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------------------
# Load the dataset from a JSON file
train_path = "complete.json"
with open(train_path) as f:
    train_ds = json.load(f)

# Create a list to hold all training instances from selected tasks
all_instances = []
# Iterate over each task in the training dataset
for task, instances in train_ds["train"].items():
    # Filter to include only specific tasks
    if task in ["artery_cls", "catheter_det", "severity_cls", "health_cls", "cadica_det", "syntax_det"]:
        for instance in instances:
            # Append each instance with a full image path and corresponding prefix and suffix
            all_instances.append({
                "image": os.path.join("/home/maxim.popov/paligemma/thesis/paligemma_dataset/train", task, instance["image"]),
                "prefix": instance["prefix"],
                "suffix": instance["suffix"]
            })

# Shuffle the instances to randomize the order
from random import shuffle
import random
random.seed(42)
shuffle(all_instances)

# -----------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------
# Set up training arguments using Hugging Face's TrainingArguments
args = TrainingArguments(
    num_train_epochs=5,                    # Number of training epochs
    remove_unused_columns=False,           # Retain all columns in the dataset
    per_device_train_batch_size=1,         # Batch size per GPU
    gradient_accumulation_steps=4,         # Accumulate gradients to simulate a larger batch size
    warmup_steps=2,                        # Warmup steps for learning rate scheduling
    learning_rate=2e-5,                    # Learning rate for the optimizer
    weight_decay=1e-6,                     # Weight decay to regularize the model
    adam_beta2=0.999,                      # Second beta parameter for the Adam optimizer
    logging_steps=100,                     # Frequency (in steps) to log training metrics
    optim="adamw_torch",                   # Optimizer choice; alternatives like paged_adamw_8bit can be used for QLoRA
    save_steps=100,                        # Save model checkpoint every 100 steps
    save_total_limit=1,                    # Limit the total number of saved checkpoints to 1
    output_dir=f"/l/users/maxim.popov/cardiogpt/paligemma_textqa_random_5ep",  # Directory for saving model checkpoints and outputs
    bf16=True,                             # Use bfloat16 precision if supported by your hardware
    report_to=["tensorboard"],             # Log training metrics to TensorBoard
    dataloader_pin_memory=False,           # Option to pin memory in DataLoader (set True if needed)
    # max_steps=2000,                      # Uncomment to set a maximum number of training steps
    save_strategy="best",                  # Save only the best model based on evaluation metrics
    metric_for_best_model="loss"           # Metric to determine the best model (lower loss is better)
    # Uncomment below to enable evaluation during training:
    # do_eval=True,
    # eval_strategy="epoch"
)

# -----------------------------------------------------------
# Data Collation Function
# -----------------------------------------------------------
# Define a custom function to collate and process each batch of data
def collate_fn(examples):
    # Prepend a special token to each prefix text
    texts = ["<image>" + x["prefix"] for x in examples]
    # Open and convert each image to RGB format using PIL
    images = [Image.open(x["image"]).convert("RGB") for x in examples]
    # Extract suffix labels from the examples
    labels = [x["suffix"] for x in examples]
    # Process text and images through the processor to obtain tensors
    # Apply padding and truncation to maintain a consistent sequence length (256 tokens)
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    ).to(DTYPE).to("cuda")
    return tokens

# -----------------------------------------------------------
# Trainer Initialization and Model Training
# -----------------------------------------------------------
# Initialize the Hugging Face Trainer with model, dataset, and training configurations
trainer = Trainer(
    model=model,                 # The model to fine-tune
    train_dataset=all_instances, # The training dataset prepared above
    # eval_dataset=eval_ds,       # Uncomment and define eval_ds for evaluation
    data_collator=collate_fn,    # Custom function to collate batch data
    args=args                    # Training arguments defined above
)

# Start the training process
trainer.train()

# After training, push the fine-tuned model to the Hugging Face Hub for sharing
trainer.push_to_hub()