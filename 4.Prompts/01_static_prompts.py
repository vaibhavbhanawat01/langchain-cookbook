from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# static prompt

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 20)

chat_model = ChatHuggingFace(llm = llm)

response = chat_model.invoke("Summarize word2vec paper in 3 lines")
print(response.content)