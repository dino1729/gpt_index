import os
from urllib.parse import parse_qs, urlparse

import gradio as gr
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

def download_ytvideo(yturl):
    if yturl:
        yt = YouTube(yturl)
        query_params = parse_qs(urlparse(yt).query)
        video_name = query_params['v'][0]
        video_name_display = f'<h2>Video Name: {video_name}</h2>'
        yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(UPLOAD_FOLDER, filename="video.mp4")
        message = 'Youtube video downloaded successfully!!'
        return message, video_name_display
    else:
        message = 'Please enter a valid Youtube URL'
        return message, ""

def build_index():
    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk("data/index.json")
    message = 'Index built successfully! You can now ask questions!!'
    return message

def ask(question):
    index = GPTSimpleVectorIndex.load_from_disk("data/index.json")
    response = index.query(question)
    return response

gradio_interface = gr.Interface(
    fn=ask,
    inputs=gr.inputs.Textbox(label="Question"),
    outputs=gr.outputs.Textbox(label="Answer"),
    title="GPT Index Q&A",
    description="Ask any question and GPT will find the closest matching answer from the indexed data.",
    examples=[
        ['How to upload an image?'],
        ['What file formats are allowed?'],
        ['What is the purpose of this tool?']
    ],
    live=True,
)

if __name__ == '__main__':
    gradio_interface.launch()
