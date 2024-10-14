from flask import Flask, request, render_template
from converter import convert_to_tensor, question_to_tensor
from generator import generate_answer_for_question
from responser import format_answer
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    question = request.form['question']

    if file and question:
        # Ensure the uploads directory exists
        os.makedirs('./uploads', exist_ok=True)
        
        file_path = f"./uploads/{file.filename}"
        file.save(file_path)

        # Convert file and question to tensor
        file_tensor = convert_to_tensor(file_path)
        question_tensor = question_to_tensor(question)

        # Generate answer
        answer = generate_answer_for_question(file_tensor, question_tensor)
        formatted_answer = format_answer(answer)

        return render_template('result.html', answer=formatted_answer)
    else:
        return "No file or question provided", 400

if __name__ == '__main__':
    app.run(debug=True)