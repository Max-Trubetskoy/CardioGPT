# CardioGPT

This repository contains code for fine-tuning and evaluating the [PaliGemma](https://huggingface.co/google/paligemma2-3b-pt-224) model using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. The code is designed to handle multiple medical image analysis tasks, including classification and detection, using both text and image inputs.

## Overview

The repository provides scripts to:
- **Fine-tune** the PaliGemma model using LoRA.
- **Evaluate** the model on various tasks, such as:
  - **Health Classification:** Determine if a vessel is healthy or diseased.
  - **Artery Classification:** Identify the artery in focus (RCA, LAD, LCX).
  - **Severity Classification:** Assess the severity of a disease.
  - **Detection Tasks:** Perform catheter, cadica (stenosis), and syntax detection using bounding box extraction and IoU metrics.
- **Parse Model Responses:** Extract bounding box coordinates from text outputs using regular expressions.
- **Compute Metrics:** Generate confusion matrices and calculate precision, recall, accuracy, and detection metrics.

## Repository Structure
├── README.md # This file\
├── test_captioning.py # Compute ROUGE and BLEU scores for the captioning\
├── test_qualitative.py # Compute Accuracy and AUC scores for quantitative tasks\
└── train_X.py # Train the model using pre-defined scenarios

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [PEFT](https://github.com/huggingface/peft)
- [Supervision](https://github.com/roboflow-ai/supervision) (or your custom supervision module)
- [Pillow](https://python-pillow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [numpy](https://numpy.org/)

### Installation

Install the required dependencies with the following:

```
pip install torch transformers peft supervision pillow scikit-learn NumPy
```

### Customization
- **Model Configuration**:
  - Change the model_id variable in the scripts to use a different pre-trained model.

- **Dataset**:
  - Modify complete.json or update the script file path to point to your dataset.

- **Tasks**:
  - The tasks list (e.g., health_cls, artery_cls, etc.) can be customized to suit your specific evaluation needs.

- **Evaluation Prompts**:
  - Update the prompt strings in the evaluation sections to tailor them to your task descriptions.
