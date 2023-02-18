import os
from urllib.parse import parse_qs, urlparse

from flask import Flask, redirect, render_template, request, url_for
from IPython.display import Markdown, display
from langchain import OpenAI
from langchain.agents import initialize_agent
from pytube import YouTube

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

BUILD_MODE = False

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAIAPIKEY")

app = Flask(__name__)

UPLOAD_FOLDER = './data' # set the upload folder path
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mkv', 'mov'} # set the allowed file extensions

#If the UPLOAD_FOLDER path does not exist, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

#Function to upload the file and display a success message within the html page
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        message = filename + ' uploaded successfully!!'
        return redirect(url_for('upload_form', msg=message))
    else:
        message = 'Allowed file types are pdf, png, jpg, jpeg, gif, mp4, avi, mkv, mov'
        return redirect(url_for('upload_form', msg=message))

#Function to download the Youtube video and display a success message within the html page
@app.route('/ytvideo', methods=['POST'])
def download_ytvideo():
    #If there is a url in the input field, download the video
    if request.form['yturl']:
        yt = YouTube(request.form['yturl'])
        query_params = parse_qs(urlparse(yt).query)
        video_name = query_params['v'][0]
        video_name_display = f'<h2>Video Name: {video_name}</h2>'
        yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(app.config['UPLOAD_FOLDER'], filename="video.mp4")
        message2 = 'Youtube video downloaded successfully!!'
        return redirect(url_for('upload_form', msg2=message2, video_name_display=video_name_display))
    else:
        message2 = 'Please enter a valid Youtube URL'
        return redirect(url_for('upload_form', msg2=message2))

#Function to build the index and return a success message within the html page
@app.route('/analyze', methods=['POST'])
def build_index():
    documents = SimpleDirectoryReader('data').load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk("data/index.json")
    message3 = 'Index built successfully! You can now ask questions!!'
    return redirect(url_for('upload_form', msg3=message3))

#Funtion to take the user quesion from the html page and return the answer once the user clicks Ask button and display it in the Answer field
@app.route('/ask', methods=['POST'])
def ask():
    index = GPTSimpleVectorIndex.load_from_disk("data/index.json")
    user_query = request.form['question']
    response = index.query(user_query)
    return render_template('index.html', msg='Answer', answer=response)

if __name__ == '__main__':
    app.run(debug=True)