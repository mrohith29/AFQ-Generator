import torch
from extractors import pdf_extractor, txt_extractor, ppt_extractor
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def convert_to_tensor(file_path):
    if file_path.endswith('.pdf'):
        text = pdf_extractor.extract_text(file_path)
    elif file_path.endswith('.txt'):
        text = txt_extractor.extract_text(file_path)
    elif file_path.endswith('.ppt'):
        text = ppt_extractor.extract_text(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Preprocess text (e.g., lowercasing, cleaning)
    text = text.lower().strip()

    # Convert text to tokens using tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=True)
    tensor = torch.tensor(tokens)
    return tensor

def question_to_tensor(question):
    # Preprocess question similarly
    question = question.lower().strip()
    tokens = tokenizer.encode(question, add_special_tokens=True)
    tensor = torch.tensor(tokens)
    return tensor
