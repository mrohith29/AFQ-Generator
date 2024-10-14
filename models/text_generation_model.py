from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

def generate_answer(tensor, question_tensor):
    # Convert tensors back to strings
    context_text = tokenizer.decode(tensor.tolist(), skip_special_tokens=True)
    question_text = tokenizer.decode(question_tensor.tolist(), skip_special_tokens=True)

    # Encode the inputs properly with truncation
    inputs = tokenizer.encode_plus(
        question_text,
        context_text,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=512,    # Limit to BERT's maximum sequence length
        truncation=True    # Truncate if it exceeds the limit
    )

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract start and end scores
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Identify the start and end positions of the answer span
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1

    # Decode the answer tokens to a string
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end], skip_special_tokens=True)
    return answer
