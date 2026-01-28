from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 100)
chat_model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(template = "Write a detail report on {topic}", input_variables=['topic'])

template2 = PromptTemplate(template = "Write a 5 line summary on the following text. /n {text}", input_variables=['text'])

parser = StrOutputParser()  

# parser use to parse llm text output to string

chain = template1 | chat_model | parser | template2 | chat_model | parser

response = chain.invoke({
    'topic': 'black hole'
})

print(response)
 