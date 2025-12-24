# CreditNLP LoRA Adapter

This directory contains the configuration for the CreditNLP LoRA adapter trained on Mistral-7B-Instruct-v0.3.

## Model Weights

The trained LoRA adapter weights (~80MB) are hosted on HuggingFace Hub:

**🤗 HuggingFace:** [pmcavallo/creditnlp-lora](https://huggingface.co/pmcavallo/creditnlp-lora)

> **Note:** If the HuggingFace link is not yet active, you can train your own adapter using the `notebooks/CreditNLP_FineTuning.ipynb` notebook on Google Colab (free T4 GPU, ~41 minutes).

## Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Quantization config for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapter from HuggingFace
model = PeftModel.from_pretrained(
    base_model, 
    "pmcavallo/creditnlp-lora"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```

## Inference

```python
def predict_default_risk(application_text: str) -> str:
    prompt = f"""Analyze this startup loan application and classify the default risk.
Respond with ONLY 'DEFAULT' or 'NO_DEFAULT'.

Application:
{application_text}

Classification:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Classification:")[-1].strip()
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Mistral-7B-Instruct-v0.3 |
| Method | QLoRA (4-bit quantization + LoRA) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training Samples | 400 |
| Epochs | 3 |
| Training Time | 41 minutes (Google Colab T4) |

## Files in This Directory

- `adapter_config.json` - LoRA configuration (rank, alpha, target modules)
- `README.md` - This file

## Requirements

```
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
```
