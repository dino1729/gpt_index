import os

from IPython.display import Markdown, display
from pytube import YouTube

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAIAPIKEY")
# download a youtube video, if it doesn't already exist, using pytube from a url and place in data folder and save it as video.mp4
if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("data/video.mp4"):
    # yt = YouTube("https://www.youtube.com/watch?v=OYk5g88PreQ")
    yt = YouTube("https://www.youtube.com/watch?v=UF8uR6Z6KLc")
    yt.streams.filter(progressive=True, file_extension="mp4").order_by(
        "resolution"
    ).desc().first().download("data", filename="video.mp4")

documents = SimpleDirectoryReader("data").load_data()
index = GPTSimpleVectorIndex(documents)

# response = index.query("Which player came on as a substitute for Chelsea in the match?")
# print(response)
# print("----------------------------------")
# response = index.query("Who won the match?")
# print(response)

response = index.query("What are the three stories from the speaker's life?")
print(response)

print("----------------------------------")

response = index.query(
    "Who and why did she refuse to sign the speaker's final adoption papers?"
)
print(response)

# save to disk
index.save_to_disk("index.json")
# load from disk
index = GPTSimpleVectorIndex.load_from_disk("index.json")
