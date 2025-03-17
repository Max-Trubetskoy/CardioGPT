import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

model_id ="google/paligemma2-3b-pt-224" # or your favorite PaliGemma
processor = PaliGemmaProcessor.from_pretrained(model_id)
device = "cuda"

# lora_config = LoraConfig(
#     r=64,
#     target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
# )

model = PaliGemmaForConditionalGeneration.from_pretrained("/l/users/maxim.popov/cardiogpt/paligemma_textqa_random_5ep/", device_map="cuda",attn_implementation='eager')#, quantization_config=bnb_config)
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
#trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344
DTYPE = model.dtype



dataset = pd.read_excel("/home/maxim.popov/paligemma/thesis/paligemma_dataset/cadica_syntax/captions_cadica.xlsx")

data_dict = {"train": [], "test": []}
for row in dataset.iterrows():
    row = row[1]
    path = row["path"].split("\\")[-1]
    if os.path.exists("/home/maxim.popov/paligemma/train/cadica_det/"+path):
        data_dict["train"].append({
            "path": "/home/maxim.popov/paligemma/train/cadica_det/"+path,
            "caption": row["caption"]
        })
    if os.path.exists("/home/maxim.popov/paligemma/test/cadica_det/"+path):
        data_dict["test"].append({
            "path": "/home/maxim.popov/paligemma/test/cadica_det/"+path,
            "caption": row["caption"]
        })


def compute_bleu(reference, candidate):
    """
    Compute BLEU score between a reference and a candidate sentence.
    """
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    return score

def compute_rouge(reference, candidate):
    """
    Compute ROUGE scores between a reference and a candidate sentence.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores


scores = {"bleu": [], "rouge": []}

for sample in tqdm(data_dict["test"][:10:2]):
    prompt = "<image>summarize this case"
    image_path = sample["path"]
    reference = sample["caption"].replace("“", "").replace("”", "")

    raw_image = Image.open(image_path)

    inputs = processor(text=prompt, images=raw_image.convert("RGB"), return_tensors="pt").to('cuda').to(torch.half)
    output = model.generate(**inputs, max_new_tokens=256, use_cache=False)
    response = processor.decode(output[0], skip_special_tokens=True).split("\n")[-1].replace("“", "").replace("”", "")

    bleu_score = compute_bleu(reference, response)
    rouge_scores = compute_rouge(reference, response)
    scores["bleu"].append(bleu_score)
    scores["rouge"].append(rouge_scores)
    sample["predict"] = response

print("BLEU Score:", np.array(scores["bleu"]).mean())
print("ROUGEL Score:", np.array([score['rougeL'].fmeasure for score in scores['rouge']]).mean())
print("ROUGE1 Score:", np.array([score['rouge1'].fmeasure for score in scores['rouge']]).mean())
