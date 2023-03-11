# try using GPT List Index!
from langchain import OpenAI
from langchain.agents import initialize_agent
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader
import os

BUILD_MODE = True

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")


# Either build or load the document index
if BUILD_MODE:
    # Put the paper you want to study into data folder
    documents = SimpleDirectoryReader('data').load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk("bostrom_paper_index.json")
else:
    index = GPTSimpleVectorIndex.load_from_disk("bostrom_paper_index.json")

# Start an interface to query GPT3 based on the index
while True:
    # Get user input for query
    user_query = input("Enter your question: ")
    response = index.query(user_query)
    print(response)
