from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents = [
    "Delhi is the capital of India",
    "Kolkata is capital of west bengal",
    "Pair is capital of Frame"]

embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=32)
result = embedding_model.embed_documents(documents)
print(result) 

