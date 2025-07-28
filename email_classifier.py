import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

torch.cuda.is_available = lambda: False
device = torch.device("cpu")

# Label names 
category_label_names = ["Finance", "HR", "Legal", "Admin"]
priority_label_names = ["High", "Medium", "Low"]

# Load fine-tuned LoRA models and tokenizers
def load_models(category_model_path="./fine_tuned_minilm_category/checkpoint-1000", priority_model_path="./fine_tuned_minilm_priority/checkpoint-800"):
   
    try:
        # Base model for both category and priority
        base_model_name = "sentence-transformers/all-MiniLM-L12-v2"

        # Load category model
        peft_config_cat = PeftConfig.from_pretrained(category_model_path)
        model_cat = AutoModelForSequenceClassification.from_pretrained(
            peft_config_cat.base_model_name_or_path,
            num_labels=len(category_label_names)
        ).to(device)
        model_cat = PeftModel.from_pretrained(model_cat, category_model_path).to(device)
        tokenizer_cat = AutoTokenizer.from_pretrained(peft_config_cat.base_model_name_or_path)

        # Load priority model
        peft_config_prio = PeftConfig.from_pretrained(priority_model_path)
        model_prio = AutoModelForSequenceClassification.from_pretrained(
            peft_config_prio.base_model_name_or_path,
            num_labels=len(priority_label_names)
        ).to(device)
        model_prio = PeftModel.from_pretrained(model_prio, priority_model_path).to(device)
        tokenizer_prio = AutoTokenizer.from_pretrained(peft_config_prio.base_model_name_or_path)

        return model_cat, tokenizer_cat, model_prio, tokenizer_prio
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None

# Clean email 
def clean_email(text):
    
    subject_match = re.search(r'Subject: (.*?)\\n', text, re.IGNORECASE)
    subject = subject_match.group(1) if subject_match else ''
    text = re.sub(r'From:.*\\n|To:.*\\n|Subject:.*\\n|Message-ID:.*\\n|Date:.*\\n', '', text)
    text = re.sub(r'-{2,}.*?-{2,}', '', text)
    text = re.sub(r'http[s]?://\\S+', '', text)
    return text.strip(), subject

# Inference functions 
def predict_category(email_text, model, tokenizer):
    
    if model is None or tokenizer is None:
        return "Error: Category model or tokenizer not loaded"
    text, subject = clean_email(email_text)
    input_text = f"[SUBJECT] {subject} [BODY] {text}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return category_label_names[predicted_class]

def predict_priority(email_text, model, tokenizer):
    
    if model is None or tokenizer is None:
        return "Error: Priority model or tokenizer not loaded"
    text, subject = clean_email(email_text)
    input_text = f"[SUBJECT] {subject} [BODY] {text}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return priority_label_names[predicted_class]