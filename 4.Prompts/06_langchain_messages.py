from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

messages = [
    SystemMessage(content = "You are helpful Assistant"),
    HumanMessage(content = "Tell me about Langchain")
]

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 20)

chat_model = ChatHuggingFace(llm = llm)
aiResponse = chat_model.invoke(messages)
aiMessage = AIMessage(content = aiResponse.content)
messages.append(aiMessage)
print(messages)
