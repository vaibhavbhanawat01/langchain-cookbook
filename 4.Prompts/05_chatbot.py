from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

# Simple chat bot using Messages

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 100)

chat_model = ChatHuggingFace(llm = llm)

chat_history = [SystemMessage(content = "You are helpful Assistant")]

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    chat_history.append(HumanMessage(content = user_input))
    aiResponse = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content = aiResponse.content))
    print("AI: ", aiResponse.content)
print(chat_history)


