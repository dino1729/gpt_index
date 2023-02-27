import json
import os
from shutil import copyfileobj
from urllib.parse import parse_qs, urlparse

import gradio as gr
import PyPDF2
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from langchain import OpenAI
from langchain.agents import initialize_agent
from newspaper import Article
from PIL import Image
from pytube import YouTube

from gpt_index import Document, GPTSimpleVectorIndex, SimpleDirectoryReader

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAIAPIKEY")
UPLOAD_FOLDER = './data' # set the upload folder path
example_queries = [["Generate key 5 point summary"], ["What are 5 main ideas of this article?"], ["What are the key lessons learned and insights in this video?"], ["List key insights and lessons learned from the paper"], ["What are the key takeaways from this article?"]]
example_qs = []

#If the UPLOAD_FOLDER path does not exist, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def pdftotext(file_name):
    """
    Function to extract text from .pdf format files
    """
    text = []
    # Open the PDF file in read-binary mode
    with open(file_name, 'rb') as file:
        # Create a PDF object
        pdf = PyPDF2.PdfReader(file)
        # Get the number of pages in the PDF document
        num_pages = len(pdf.pages)
        # Iterate over every page
        for page in range(num_pages):
            # Extract the text from the page
            result = pdf.pages[page].extract_text()
            text.append(result)
    text = "\n".join(text)
    return text

def preprocesstext(text):
    """
    Function to preprocess text
    """
    # Split the string into lines
    lines = text.splitlines()
    # Use a list comprehension to filter out empty lines
    lines = [line for line in lines if line.strip()]
    # Join the modified lines back into a single string
    text = '\n'.join(lines)
    return text

def processfiles(files):
    """
    Function to extract text from documents
    """
    textlist = []
    # Iterate over provided files
    for file in files:
        # Get file name
        file_name = file.name
        # Get extention of file name
        ext = file_name.split(".")[-1].lower()
        text = ""
        # Process document based on extention
        if ext == "pdf":
            text = pdftotext(file_name)
        # Preprocess text
        text = preprocesstext(text)
        # Append the text to final result
        textlist.append(text)
    return textlist

def fileformatvaliditycheck(files):
    #Function to check validity of file formats
    for file in files:
        file_name = file.name
        # Get extention of file name
        ext = file_name.split(".")[-1].lower()
        if ext not in ["pdf", "txt", "docx", "png", "jpg", "jpeg"]:
            return False
    return True

def createdocumentlist(files):
    documents = []
    for file in files:
        documents.append(Document(file))
    return documents

def savetodisk(files):
    #Save the files to the UPLOAD_FOLDER
    for file in files:
        #Extract the file name
        filename_with_path = file.name
        file_name = file.name.split("/")[-1]
        #Open the file in read-binary mode
        with open(filename_with_path, 'rb') as f:
            #Save the file to the UPLOAD_FOLDER
            with open(UPLOAD_FOLDER + "/" + file_name, 'wb') as f1:
                copyfileobj(f, f1)

def build_index():

    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    index = GPTSimpleVectorIndex(documents) 
    index.save_to_disk(UPLOAD_FOLDER + "/index.json")


def clearnonfiles(files):
    #Ensure the UPLOAD_FOLDER contains only the files uploaded
    for file in os.listdir(UPLOAD_FOLDER):
        if file not in [file.name.split("/")[-1] for file in files]:
            os.remove(UPLOAD_FOLDER + "/" + file)

def clearnonvideos():
    #Ensure the UPLOAD_FOLDER contains only the video downloaded
    for file in os.listdir(UPLOAD_FOLDER):
        if file not in ["video.mp4"]:
            os.remove(UPLOAD_FOLDER + "/" + file)

def clearnonarticles():
    #Ensure the UPLOAD_FOLDER contains only the article downloaded
    for file in os.listdir(UPLOAD_FOLDER):
        if file not in ["article.txt"]:
            os.remove(UPLOAD_FOLDER + "/" + file)

def upload_file(files):

    global example_queries
    #Basic checks
    if not files:
        return "Please upload a file before proceeding",gr.Dataset.update(samples=example_queries)
    
    fileformatvalidity = fileformatvaliditycheck(files)
    #Check if all the files are in the correct format
    if not fileformatvalidity:
        return "Please upload documents in pdf/txt/docx/png/jpg/jpeg format only.",gr.Dataset.update(samples=example_queries)
    
    #Save files to UPLOAD_FOLDER
    savetodisk(files)
    #Clear files from UPLOAD_FOLDER
    clearnonfiles(files)
    #Build index
    build_index()
    #Generate example queries
    example_queries = example_generator()

    return "Files uploaded and Index built successfully!",gr.Dataset.update(samples=example_queries)

def download_ytvideo(url):

    global example_queries
    #If there is a url in the input field, download the video
    if url:
        yt = YouTube(url)
        yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(UPLOAD_FOLDER, filename="video.mp4")
        #Clear files from UPLOAD_FOLDER
        clearnonvideos()
        #Build index
        build_index()
        #Generate example queries
        example_queries = example_generator()

        return "Youtube video downloaded and Index built successfully!",gr.Dataset.update(samples=example_queries)
    else:
        return "Please enter a valid Youtube URL",gr.Dataset.update(samples=example_queries)

def download_art(url):

    global example_queries
    #If there is a url in the input field, download the article
    if url:
        #Extract the article
        article = Article(url)
        article.download()
        article.parse()
        #Save the article to the UPLOAD_FOLDER
        with open(UPLOAD_FOLDER + "/article.txt", 'w') as f:
            f.write(article.text)
        #Clear files from UPLOAD_FOLDER
        clearnonarticles()
        #Build index
        build_index()
        #Generate example queries
        example_queries = example_generator()

        return "Article downloaded and Index built successfully!",gr.Dataset.update(samples=example_queries)
    else:
        return "Please enter a valid URL",gr.Dataset.update(samples=example_queries)

def ask(question,history):
    history = history or []
    s = list(sum(history, ()))
    s.append(question)
    inp = ' '.join(s)

    index = GPTSimpleVectorIndex.load_from_disk(UPLOAD_FOLDER + "/index.json")
    response = index.query(question)
    answer = response.response

    history.append((question, answer))

    return history, history

def ask_query(question):

    index = GPTSimpleVectorIndex.load_from_disk(UPLOAD_FOLDER + "/index.json")
    response = index.query(question)
    answer = response.response

    return answer

def cleartext(query, output):
  #Function to clear text
  return ["", ""]

def example_generator():
    global example_queries, example_qs
    try:
        example_qs = [[str(item)] for item in eval(ask_query("Generate the top 5 relevant questions from the input paragraph. Output must be must in the form of python list of 5 strings.").replace('\n', ''))]
    except:
        example_qs = example_queries
    return example_qs

def update_examples():
    global example_queries
    example_queries = example_generator()
    return gr.Dataset.update(samples=example_queries)

def load_example(example_id):
    global example_queries
    return example_queries[example_id][0]

with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    gr.Markdown(
        """
        <h1><center><b>LLM Dino Bot</center></h1>
        """
    )
    gr.Markdown(
        """
        This app uses the Transformer magic to answer questions about the content of a video, article or document. Either upload a document or enter a Youtube/Article URL to get started.
        """
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            with gr.Box():
                files = gr.File(label = "Upload the files to be analyzed", file_count="multiple")
                with gr.Row():
                    upload_button = gr.Button("Upload").style(full_width=False)
                    upload_output = gr.Textbox(label="Upload Status")
            with gr.Tab(label="Video Analyzer"):
                yturl = gr.Textbox(placeholder="Input must be a URL", label="Enter Youtube URL")
                with gr.Row():
                    download_button = gr.Button("Download").style(full_width=False)
                    download_output = gr.Textbox(label="Video download Status")
            with gr.Tab(label="Article Analyzer"):
                arturl = gr.Textbox(placeholder="Input must be a URL", label="Enter Article URL")
                with gr.Row():
                    adownload_button = gr.Button("Download").style(full_width=False)
                    adownload_output = gr.Textbox(label="Article download Status")
        with gr.Column(scale=2, min_width=650):
            with gr.Box():
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLM Bot").style(color_map=["blue","grey"])
                state = gr.State([])
                with gr.Row():
                    query = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
                    submit_button = gr.Button("Ask").style(full_width=False)
                    clearquery_button = gr.Button("Clear").style(full_width=False)
                examples = gr.Dataset(samples=example_queries, components=[query], type="index")
                submit_button.click(ask, inputs=[query, state], outputs=[chatbot, state])
                query.submit(ask, inputs=[query, state], outputs=[chatbot, state])
            clearchat_button = gr.Button("Clear Chat")

    # Upload button for uploading files
    upload_button.click(upload_file, inputs=[files], outputs=[upload_output,examples], show_progress=True)
    # Download button for downloading youtube video
    download_button.click(download_ytvideo, inputs=[yturl], outputs=[download_output,examples], show_progress=True)
    # Download button for downloading article
    adownload_button.click(download_art, inputs=[arturl], outputs=[adownload_output,examples], show_progress=True)

    #Load example queries
    examples.click(load_example, inputs=[examples], outputs=[query])

    clearquery_button.click(cleartext, inputs=[query,query], outputs=[query,query])
    clearchat_button.click(cleartext, inputs=[query,chatbot], outputs=[query,chatbot])

    live = True

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0',server_port=7860,debug=True)