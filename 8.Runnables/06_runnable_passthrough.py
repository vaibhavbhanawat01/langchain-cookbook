from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

template1 = PromptTemplate(template = 'Write a joke about {topic}', 
                           input_variables=['topic'])
template2 = PromptTemplate(template='Give me explaination of Joke {joke}')

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_chain = RunnableSequence(template1, chat_model, parser)
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination' : RunnableSequence(template2, chat_model, parser)
})
chain = RunnableSequence(joke_chain, parallel_chain)
response = chain.invoke({'topic', 'American People'})
print(response)