from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

load_dotenv()


template1 = PromptTemplate(template = 'Write a joke about {topic}', 
                           input_variables=['topic'])

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_chain = RunnableSequence(template1, chat_model, parser)

def countWord(joke):
    return len(joke.split())

# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'count': RunnableLambda(countWord)
# })

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'count': RunnableLambda(lambda x: len(x.split()))
})

chain = RunnableSequence(joke_chain, parallel_chain)
response = chain.invoke({'topic', 'cricket'})
print(response)