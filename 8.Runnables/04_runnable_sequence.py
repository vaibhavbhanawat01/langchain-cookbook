from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

#Sequence chaining using RunnableSequence

template1 = PromptTemplate(template = 'Write a joke about {topic}', 
                           input_variables=['topic'])
template2 = PromptTemplate(template='Give me explaination of Joke {joke}')

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableSequence(template1, chat_model, parser, template2, chat_model, parser)
print(chain.invoke('Indian'))