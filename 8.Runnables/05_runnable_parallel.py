from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

# Parallel chaining using RunnableParallel

template1 = PromptTemplate(template='Write a tweet about {topic}',
                           input_variables=['topic'])
template2 = PromptTemplate(template='Write a LinkedIn post about {topic}',
                           input_variables=['topic'])
llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(template1, chat_model, parser),
    'post' : RunnableSequence(template2, chat_model, parser)
})

response = parallel_chain.invoke({'topic': 'India'})
print(response)