import os

from IPython.display import Markdown, display
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from pytube import YouTube

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAIAPIKEY")
# download a youtube video, if it doesn't already exist, using pytube from a url and place in data folder and save it as video.mp4
if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("data/video.mp4"):
    # Get YouTube URL from user
    yt = YouTube(input("Enter YouTube URL: "))
    yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download("data", filename="video.mp4")

documents = SimpleDirectoryReader("data").load_data()
index = GPTSimpleVectorIndex(documents)

# Get the guestion from user
question1 = input("Enter your first question: ")
response = index.query(question1)
print(response)

print("----------------------------------")

question2 = input("Enter your second question: ")
response = index.query(question2)
print(response)

# save to disk
index.save_to_disk("index.json")
# load from disk
index = GPTSimpleVectorIndex.load_from_disk("index.json")
