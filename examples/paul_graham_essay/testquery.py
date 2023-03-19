import os

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

# Get API key from environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
os.environ["OPENAI_API_BASE"] = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")
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

# max LLM token input size
max_input_size = 500
# set number of output tokens
num_output = 48
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
embedding_llm = LangchainEmbedding(OpenAIEmbeddings(chunk_size=1))

documents = SimpleDirectoryReader("data").load_data()
index = GPTSimpleVectorIndex(documents, embed_model= embedding_llm, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

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
