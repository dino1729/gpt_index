import os

import openai
from IPython.display import Markdown, display
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    LangchainEmbedding,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,
)

os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
os.environ["OPENAI_API_BASE"] = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_type = "azure"
openai.api_version = "2022-12-01"


llm = AzureOpenAI(deployment_name="text-davinci-003")
llm_predictor = LLMPredictor(llm=llm)

embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
    document_model_name="text-embedding-ada-002",
    query_model_name="text-embedding-ada-002"
))

# max LLM token input size
max_input_size = 500
# set number of output tokens
num_output = 48
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

documents = SimpleDirectoryReader("data").load_data()
index = GPTSimpleVectorIndex(documents, embed_model=embedding_llm, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

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
