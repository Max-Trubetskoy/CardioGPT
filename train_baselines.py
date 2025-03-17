# Import required libraries and modules
import json                             # For handling JSON files
import numpy as np                      # For numerical operations
from PIL import Image                   # For image processing

# Import specific classes from transformers and PEFT
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig  # For potential quantization (currently not used)
from peft import get_peft_model, LoraConfig   # For parameter-efficient fine-tuning using LoRA
import torch                            # For PyTorch operations
from transformers import TrainingArguments, Trainer  # For training configuration and training process

# Optionally, log in to Hugging Face Hub (uncomment if required)

# Load the training dataset from a JSON file
train_path = "complete.json"
with open(train_path) as f:
    train_ds = json.load(f)

# Loop over each task in the training dataset.
# Each key in train_ds['train'] represents a distinct task.
for task in list(train_ds["train"].keys()):
    
    # -------------------------------
    # Model and Processor Initialization
    # -------------------------------
    
    # Define the model ID (you can replace this with your preferred model)
    model_id = "google/paligemma2-3b-pt-224"  # or your favorite PaliGemma model
    
    # Initialize the processor from the pretrained model
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    # Set device to CUDA for GPU acceleration
    device = "cuda"
    
    # Load the pre-trained model with half precision and move it to GPU
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.half  # Use half precision for faster computation and reduced memory usage
    ).to(device)
    
    # Disable cache to improve training stability
    model.config.use_cache = False
    
    # Optional: Configure 4-bit quantization 
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    # -------------------------------
    # LoRA Configuration and Model Preparation
    # -------------------------------
    
    # Define the LoRA configuration for parameter-efficient fine-tuning (PEFT)
    lora_config = LoraConfig(
        r=8,  # Rank of the LoRA matrices (controls the size of the additional trainable parameters)
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",  # Task type for causal language modeling
    )
    
    # Reload the model with device mapping and custom attention implementation.
    # You can add the quantization configuration (bnb_config) if needed.
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        attn_implementation="eager"
        # , quantization_config=bnb_config  # Uncomment to use quantization
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

    # Apply the LoRA configuration to the model for fine-tuning
    model = get_peft_model(model, lora_config)
    
    # Print the number of trainable parameters versus the total parameters
    model.print_trainable_parameters()
    # Expected output example:
    # trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344
    
    # Save the model's data type for later use when converting tensors
    DTYPE = model.dtype
    
    # -------------------------------
    # Data Preparation
    # -------------------------------
    
    # Retrieve the training and evaluation datasets for the current task
    task_ds = train_ds["train"][task]
    eval_ds = train_ds["test"][task]
    
    # -------------------------------
    # Training Configuration
    # -------------------------------
    
    # Define training arguments
    args = TrainingArguments(
        num_train_epochs=10,                    # Total number of training epochs
        remove_unused_columns=False,            # Keep all columns in the dataset (no column removal)
        per_device_train_batch_size=1,          # Batch size per GPU
        gradient_accumulation_steps=4,          # Gradients are accumulated over multiple steps to simulate a larger batch size
        warmup_steps=2,                         # Number of steps for the learning rate warmup
        learning_rate=2e-5,                     # Learning rate for the optimizer
        weight_decay=1e-6,                      # Weight decay to prevent overfitting
        adam_beta2=0.999,                       # Beta2 parameter for the Adam optimizer
        logging_steps=100,                      # Logging frequency (in steps)
        optim="adamw_torch",                    # Optimizer type (you can use paged optimizers for QLoRA if desired)
        save_steps=100,                         # Save checkpoint every 100 steps
        save_total_limit=1,                     # Limit the total number of saved checkpoints
        output_dir=f"/l/users/maxim.popov/paligemma/thesis_baselines/paligemma_{task}_lora8",  # Directory to save checkpoints and outputs
        bf16=True,                              # Use bfloat16 precision (if supported) for faster training
        report_to=["tensorboard"],              # Report training metrics to TensorBoard
        dataloader_pin_memory=False,            # Whether to pin memory in the dataloader (can be set True if needed)
        save_strategy="best",                   # Save the best model based on the evaluation metric
        metric_for_best_model="loss"            # Metric used to determine the best model (lower loss is better)
        # Uncomment the following lines to enable evaluation during training:
        # do_eval=True,
        # eval_strategy="epoch"
    )
    
    # -------------------------------
    # Data Collation Function
    # -------------------------------
    
    # Define a custom collate function to process and batch examples
    def collate_fn(examples):
        # Prepend a special image token to each text prefix
        texts = ["<image>" + x["prefix"] for x in examples]
        split = "train"  # Specify the data split; update if needed
        
        # Open and convert each image to RGB using PIL
        images = [Image.open(f"{split}/{task}/" + x["image"]).convert("RGB") for x in examples]
        
        # Extract text labels (suffixes) from the examples
        labels = [x["suffix"] for x in examples]
        
        # Process texts, images, and labels using the processor.
        # The processor tokenizes the text and processes the image, and returns tensors.
        # Padding is applied to ensure a consistent sequence length of 128.
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
    
    # -------------------------------
    # Training and Model Upload
    # -------------------------------
    
    # Initialize the Trainer with the model, training dataset, and other configurations
    trainer = Trainer(
        model=model,                # The model to be fine-tuned
        train_dataset=task_ds,      # Training dataset for the current task
        # eval_dataset=eval_ds,     # Uncomment to enable evaluation with the evaluation dataset
        data_collator=collate_fn,   # Custom collate function to process the inputs
        args=args                   # Training arguments defined above
    )
    
    # Start the training process
    trainer.train()
    
    # After training, push the fine-tuned model to the Hugging Face Hub for sharing
    trainer.push_to_hub()