import os
from shutil import copyfileobj
from urllib.parse import parse_qs, urlparse

import gradio as gr
import PyPDF2
from IPython.display import Markdown, display
from langchain import OpenAI
from langchain.agents import initialize_agent
from PIL import Image
from pytube import YouTube

from gpt_index import Document, GPTSimpleVectorIndex, SimpleDirectoryReader

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAIAPIKEY")
UPLOAD_FOLDER = './data' # set the upload folder path

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
    #doctextlist = processfiles(files)
    #documents = createdocumentlist(doctextlist)
    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    #index = GPTSimpleVectorIndex(documents, chunk_size_limit=1000)
    index = GPTSimpleVectorIndex(documents) 
    #Save the index to UPLOAD_FOLDER
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

def upload_file(files):
    #Basic checks
    if not files:
        return "Please upload a file before proceeding"
    
    fileformatvalidity = fileformatvaliditycheck(files)
    #Check if all the files are in the correct format
    if not fileformatvalidity:
        return "Please upload documents in pdf/txt/docx/png/jpg/jpeg format only."
    
    #Save files to UPLOAD_FOLDER
    savetodisk(files)
    #Clear files from UPLOAD_FOLDER
    clearnonfiles(files)
    #Build index
    build_index()
    return "Files uploaded and Index built successfully!"

def download_ytvideo(url):
    #If there is a url in the input field, download the video
    if url:
        yt = YouTube(url)
        yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(UPLOAD_FOLDER, filename="video.mp4")
        #Clear files from UPLOAD_FOLDER
        clearnonvideos()
        #Build index
        build_index()
        return "Youtube video downloaded and Index built successfully!"
    else:
        return "Please enter a valid Youtube URL"

def ask(question):
    index = GPTSimpleVectorIndex.load_from_disk(UPLOAD_FOLDER + "/index.json")
    #response = index.query(question, mode = "embedding", similarity_top_k = 2)
    response = index.query(question)
    answer = response.response

    #source1 = response.source_nodes[0].source_text
    #source2 = response.source_nodes[1].source_text

    return answer

def cleartext(query, output):
  """
  Function to clear text
  """
  return ["", ""]

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1><center><b>GPT Answering Bot</center></h1>
        """
    )
    gr.Markdown(
        """
        This app uses the GPT-3 API to answer questions about the content of a video or document.
        """
    )
    with gr.Row():
        with gr.Column():
            files = gr.File(label = "Upload your files here", file_count="multiple")
            upload_output = gr.Textbox(label="Upload Status")
            upload_button = gr.Button("Upload")
            yturl = gr.Textbox(label="Enter Youtube URL")
            download_button = gr.Button("Download")
            download_output = gr.Textbox(label="Download Status")
        with gr.Column():
            query = gr.Textbox(label="Enter your question here")
            submit_button = gr.Button("Submit")
            ans_output = gr.Textbox(label="Answer")
            #source1 = gr.Textbox(label="Source 1")
            #source2 = gr.Textbox(label="Source 2")
            clear_button = gr.Button("Clear")
    # Upload button for uploading files
    upload_button.click(upload_file, inputs=[files], outputs=[upload_output])
    # Download button for downloading youtube video
    download_button.click(download_ytvideo, inputs=[yturl], outputs=[download_output])
    # Submit button for submitting the query
    submit_button.click(ask, inputs=[query], outputs=[ans_output])
    # Clear button for clearing the output
    clear_button.click(cleartext, inputs=[query,ans_output], outputs=[query, ans_output])
    live = True

if __name__ == '__main__':
    demo.launch()