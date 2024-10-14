from models.text_generation_model import generate_answer

def generate_answer_for_question(tensor, question_tensor):
    return generate_answer(tensor, question_tensor)