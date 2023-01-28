import os
os.environ['OPENAI_API_KEY'] = "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"

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