# try using GPT List Index!
import json
import os
from IPython.display import Markdown, display
from pytube import YouTube
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    LangchainEmbedding,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,
)

BUILD_MODE = True

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
os.environ["OPENAI_API_BASE"] = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_type = "azure"
openai.api_version = "2022-12-01"
# max LLM token input size
max_input_size = 500
# set number of output tokens
num_output = 48
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

llm = AzureOpenAI(deployment_name="text-davinci-003", model_kwargs={
    "api_type": "azure",
    "api_version": "2022-12-01",
})
llm_predictor = LLMPredictor(llm=llm)

embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
    document_model_name="text-similarity-curie-001",
    query_model_name="text-similarity-curie-001"
))

# download a youtube video, if it doesn't already exist, using pytube from a url and place in data folder and save it as video.mp4
if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("data/video.mp4"):
    # Get YouTube URL from user
    yt = YouTube(input("Enter YouTube URL: "))
    yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download("data", filename="video.mp4")

# Either build or load the document index
if BUILD_MODE:
    # Put the paper you want to study into data folder
    documents = SimpleDirectoryReader('data').load_data()
    index = GPTSimpleVectorIndex(documents)
    #Save the index with the name video_index.json
    index.save_to_disk("video_index.json")
else:
    index = GPTSimpleVectorIndex.load_from_disk("video_index.json")

# Start an interface to query GPT3 based on the index
while True:
    # Get user input for query
    user_query = input("Enter your question: ")
    response = index.query(user_query)
    print(response)