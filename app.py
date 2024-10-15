from flask import Flask, request, render_template, redirect, url_for, session
from converter import convert_to_tensor, question_to_tensor
from generator import generate_answer_for_question
from responser import format_answer
from authlib.integrations.flask_client import OAuth
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("client_secret")  # Ensure this line reads your secret key from .env

# OAuth Configuration
oauth = OAuth(app)
github = oauth.register(
    name='github',
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    client_kwargs={'scope': 'user:email'},
    api_base_url='https://api.github.com/',
)

@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login')
def login():
    redirect_uri = url_for('authorize', _external=True)
    return github.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    token = github.authorize_access_token()
    user_info = github.get('user').json()
    session['user'] = user_info
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user' not in session:
        return redirect(url_for('login'))

    file = request.files['file']
    question = request.form['question']

    if file and question:
        # Read file directly from the stream
        file_content = file.read()

        # Process the file content in-memory using BytesIO
        file_tensor = convert_to_tensor(BytesIO(file_content))
        question_tensor = question_to_tensor(question)

        # Generate answer
        answer = generate_answer_for_question(file_tensor, question_tensor)
        formatted_answer = format_answer(answer)

        return render_template('result.html', answer=formatted_answer)
    else:
        return "No file or question provided", 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)