from langchain.tools import tool
from langchain_core.tools import InjectedToolArg # passing on tool output value to next tool
from typing import Annotated
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.messages import HumanMessage
import json

# tool for currency conversion for any two currency


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetch the currency conversion factor between base_currency and target_currency
    """
    url = f"https://v6.exchangerate-api.com/v6/0c15479f494bd15b8c13706e/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()


print(get_conversion_factor.invoke({'base_currency': 'USD', 'target_currency': 'INR'}))

@tool
def convert(base_currency_value: int, conversion_factor: Annotated[float, InjectedToolArg]) -> float:
    """
    This function will convert base currency value to target currency using conversion_factor
    """
    return base_currency_value * conversion_factor

# try with openAPI
llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 20)

chat_model = ChatHuggingFace(llm = llm)
chat_model_with_tools = chat_model.bind_tools([get_conversion_factor, convert])
messages = []
humanMessage = HumanMessage('What is the conversion factor between USD and INR, and based on that can you convert 10 USD to INR')
messages.append(humanMessage)
ai_message = chat_model_with_tools.invoke(messages)
print(ai_message)


for tool_call in ai_message.tool_calls:
    if(tool_call['name'] == 'get_conversion_factor'):
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.load(tool_message1.content)['conversion_rate']
        messages.append(tool_message1)
    if(tool_call['name'] == 'convert'):
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)
print(messages)
response = chat_model_with_tools.invoke(messages)
print(response)
