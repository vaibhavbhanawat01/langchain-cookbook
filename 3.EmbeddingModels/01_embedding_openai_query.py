from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=32)
result = embedding_model.embed_query("Delhi is the capital of India")
print(result)

