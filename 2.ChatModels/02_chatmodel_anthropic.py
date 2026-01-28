from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

anthropicModel = ChatAnthropic(model='claude-sonnet-4-5-20250929')
response = anthropicModel.invoke("What is capital of india")
print(response)
