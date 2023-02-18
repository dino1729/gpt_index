import os
from urllib.parse import parse_qs, urlparse

import gradio as gr
from IPython.display import Markdown, display
from langchain import OpenAI
from langchain.agents import initialize_agent
from pytube import YouTube

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

BUILD_MODE = False

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAIAPIKEY")

UPLOAD_FOLDER = './data' # set the upload folder path
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mkv', 'mov'} # set the allowed file extensions

#If the UPLOAD_FOLDER path does not exist, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file(file):
    if file and allowed_file(file.name):
        filename = file.name
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        message = filename + ' uploaded successfully!!'
        return message
    else:
        message = 'Allowed file types are pdf, png, jpg, jpeg, gif, mp4, avi, mkv, mov'
        return message


def download_ytvideo(url):
    #If there is a url in the input field, download the video
    if url:
        yt = YouTube(url)
        query_params = parse_qs(urlparse(yt).query)
        video_name = query_params['v'][0]
        video_name_display = f'<h2>Video Name: {video_name}</h2>'
        yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(UPLOAD_FOLDER, filename="video.mp4")
        message = 'Youtube video downloaded successfully!!'
        return message, video_name_display
    else:
        message = 'Please enter a valid Youtube URL'
        return message, ''

def build_index():
    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(os.path.join(UPLOAD_FOLDER, "index.json"))
    message = 'Index built successfully! You can now ask questions!!'
    return message


def ask(question):
    index = GPTSimpleVectorIndex.load_from_disk(os.path.join(UPLOAD_FOLDER, "index.json"))
    response = index.query(question)
    return response

# Set up Gradio interface
file_uploader = gr.inputs.File(upload_folder=UPLOAD_FOLDER)
text_input = gr.inputs.Textbox(lines=3, placeholder="Type your question here...")
url_input = gr.inputs.Textbox(label="Youtube URL")
output_text = gr.outputs.Textbox(label="Answer", default="No answer yet.")
interface = gr.Interface(upload_file, [file_uploader], output_text, capture_session=True, title="File Uploader")

interface2 = gr.Interface(download_ytvideo, [url_input], [output_text, gr.outputs.HTML(label="Video Name")], capture_session=True, title="Youtube Video Downloader")

interface3 = gr.Interface(build_index, [], output_text, capture_session=True, title="Build Index")

interface4 = gr.Interface(ask, text_input, output_text, capture_session=True, title="Ask a Question")

# Launch the app
gr.Interface([interface, interface2, interface3, interface4], gradio_server_name="my-gradio-app").launch()
