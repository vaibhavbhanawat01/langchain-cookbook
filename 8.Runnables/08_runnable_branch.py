from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch 

load_dotenv()


# use in conditional chaining 

template1 = PromptTemplate(template = 'Write a detail report on {topic}', 
                           input_variables=['topic'])

template2 = PromptTemplate(template = 'Summarize the following text {text}', 
                           input_variables=['text'])

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 100)
chat_model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

report_chain = RunnableSequence(template1, chat_model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 50, RunnableSequence(template2, chat_model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_chain, branch_chain)
response = chain.invoke({'topic', 'Cricket'})
print(response)
print(len(response.split()))