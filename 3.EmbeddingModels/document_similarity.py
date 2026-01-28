from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

model_name = 'sentence-transformers/all-mpnet-base-v2'

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

embedding  = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
query = "Tell me about virat kolhi"
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
print(doc_embeddings)
print(query_embedding)
print(type(doc_embeddings))
print(type(query_embedding))
print(cosine_similarity([query_embedding], doc_embeddings))
similarity = cosine_similarity([query_embedding], doc_embeddings)[0]
print(f"Each document similarity: {similarity}")
index, score = sorted(list(enumerate(similarity)), key = lambda x:x[1])[-1]
print(f"Simlarity Score is: {score}")
print(f"Query is matching with document : {documents[index]}")
