import os

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAIAPIKEY")

from IPython.display import Markdown, display

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = GPTSimpleVectorIndex(documents)

response = index.query("What are the best places to roam the streets of New York?")
print(response)

print("----------------------------------")

response = index.query("Who is the best mayor of New York in terms of econmic growth?")
print(response)

# save to disk
index.save_to_disk("index.json")
# load from disk
index = GPTSimpleVectorIndex.load_from_disk("index.json")
