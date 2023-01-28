import os

#Get API key from environment variable
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAIAPIKEY')

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from IPython.display import Markdown, display

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(documents)

response = index.query("What is the reason for the author's hatred towards the jews?")
print(response)

# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json')