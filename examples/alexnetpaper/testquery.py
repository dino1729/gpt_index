import os

#Get API key from environment variable
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAIAPIKEY')

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from IPython.display import Markdown, display

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(documents)

response = index.query("Summarize the paper in 3 sentences")
print(response)

print("--------------------------------------------")

response = index.query("What profound impact did this paper have in the history of deep learning research and hardware requriements?")
print(response)

# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json')