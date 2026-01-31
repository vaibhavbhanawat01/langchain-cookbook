from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.messages import HumanMessage


# try with openAPI
llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 20)

chat_model = ChatHuggingFace(llm = llm)

messages = []
query = HumanMessage('can you multiply 3 with 1000') # HumanMessage
messages.append(query)

# Tool example
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

llm_with_tools = chat_model.bind_tools([multiply])

response = llm_with_tools.invoke(messages)
messages.append(response) # AI Message
print(response)

tool_result = multiply.invoke(response.tool_calls[0])

messages.append(tool_result) # ToolMessage

result = llm_with_tools.invoke(messages)
print(result.content)
