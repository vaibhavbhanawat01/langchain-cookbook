from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 10)
chat_model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(template = "Write a detail report on {topic}", input_variables=['topic'])

template2 = PromptTemplate(template = "Write a 5 line summary on the following text. /n {text}", input_variables=['text'])


prompt1 = template1.invoke({
    'topic': 'black hole'
})

response1 = chat_model.invoke(prompt1);
print(response1.content)
prompt2 = template2.invoke({
    'text': response1.content
})

response2 = chat_model.invoke(prompt2)
print(response2.content)