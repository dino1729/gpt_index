import os

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAIAPIKEY")

from IPython.display import Markdown, display
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = GPTSimpleVectorIndex(documents)

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
