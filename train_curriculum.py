# -----------------------------------------------------------
# Import Libraries and Modules
# -----------------------------------------------------------
import json                             # For JSON file handling
from PIL import Image                  # For image processing

# Import necessary components from Hugging Face Transformers and PEFT
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig   # For potential quantization (currently not used)
from peft import get_peft_model, LoraConfig     # For applying LoRA for efficient fine-tuning
from transformers import TrainingArguments, Trainer  # For training configuration and process

# -----------------------------------------------------------
# Define Tasks and Load Dataset
# -----------------------------------------------------------
# Tuple of task names to be processed
tasks = (
    'cathno_det', 'catheter_det', 'artery_type', 'artery_cls',  
    'lesion_cls', 'cadica_det', 'kemerovo_det', 'syntax_det', 
    'arcade_det', 'severity_cls'
)

# Load the training dataset from a JSON file
train_path = "complete.json"
with open(train_path) as f:
    train_ds = json.load(f)

# -----------------------------------------------------------
# Model and Processor Setup
# -----------------------------------------------------------
# Define the model ID (replace with your preferred PaliGemma model if desired)
model_id = "google/paligemma2-3b-pt-224"  # or your favorite PaliGemma

# Initialize the processor to handle both text and image inputs
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Set the device for computation (GPU acceleration)
device = "cuda"

# Optional: Configure 4-bit quantization (currently commented out)
# bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)



# -----------------------------------------------------------
# LoRA Configuration and Model Initialization
# -----------------------------------------------------------
# Define LoRA configuration for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",  # Specify the task type (causal language modeling)
)

# Load the pre-trained model with custom attention implementation and device mapping.
# Optionally, include quantization configuration if needed.
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

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters summary for verification
model.print_trainable_parameters()
# Example output:
# trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344

# Store the model's data type for later tensor conversion
DTYPE = model.dtype

# -----------------------------------------------------------
# Training Loop for Each Task
# -----------------------------------------------------------
for task in tasks:
    
    # Retrieve training and evaluation datasets for the current task
    task_ds = train_ds["train"][task]
    eval_ds = train_ds["test"][task]  # Note: eval_ds is defined but not used in Trainer below

    # ---------------------------
    # Training Configuration
    # ---------------------------
    args = TrainingArguments(
        num_train_epochs=10,                     # Number of training epochs
        remove_unused_columns=False,             # Do not remove any columns from the dataset
        per_device_train_batch_size=1,           # Batch size per GPU
        gradient_accumulation_steps=4,           # Accumulate gradients to simulate a larger batch size
        warmup_steps=2,                          # Number of warmup steps for learning rate scheduling
        learning_rate=2e-5,                      # Learning rate for the optimizer
        weight_decay=1e-6,                       # Weight decay for regularization
        adam_beta2=0.999,                        # Second beta parameter for the Adam optimizer
        logging_steps=100,                       # Log training metrics every 100 steps
        optim="adamw_torch",                     # Optimizer type (alternative: paged optimizers for QLoRA)
        save_steps=100,                          # Save a checkpoint every 100 steps
        save_total_limit=1,                      # Limit the total number of saved checkpoints to 1
        output_dir=f"/l/users/maxim.popov/paligemma/new_baselines/paligemma_curriculum_{task}_checkpoint_lora8",
                                                 # Output directory for the current task
        bf16=True,                               # Use bfloat16 precision if supported
        report_to=["tensorboard"],               # Log training metrics to TensorBoard
        dataloader_pin_memory=False,             # Whether to pin memory in DataLoader
        # max_steps=2000,                        # Optionally, set a maximum number of training steps
        save_strategy="best",                    # Save only the best model based on evaluation metric
        metric_for_best_model="loss"             # Metric used to determine the best model (lower loss is better)
        # Uncomment the following lines to enable evaluation during training:
        # do_eval=True,
        # eval_strategy="epoch"
    )
    
    # ---------------------------
    # Define Data Collation Function
    # ---------------------------
    def collate_fn(examples):
        # Prepend a special token to each prefix text
        texts = ["<image>" + x["prefix"] for x in examples]
        split = "train"  # Data split folder; modify if necessary
        
        # Open and convert each image to RGB format using PIL
        images = [Image.open(f"{split}/{task}/" + x["image"]).convert("RGB") for x in examples]
        
        # Extract the suffix labels from the examples
        labels = [x["suffix"] for x in examples]
        
        # Process text, images, and labels using the processor
        # The output tensors are padded/truncated to a max length of 128 tokens
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

    # ---------------------------
    # Initialize Trainer and Train Model
    # ---------------------------
    trainer = Trainer(
        model=model,                # The fine-tuning model
        train_dataset=task_ds,      # Training dataset for the current task
        # eval_dataset=eval_ds,     # Uncomment to enable evaluation during training
        data_collator=collate_fn,   # Custom collate function for data processing
        args=args                   # Training arguments defined above
    )
    
    # Start training for the current task
    trainer.train()
    
    # After training, push the fine-tuned model to the Hugging Face Hub
    trainer.push_to_hub()