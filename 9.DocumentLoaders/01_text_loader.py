from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

loader = TextLoader('/Users/monika/Downloads/langchain-cookbook/9.DocumentLoaders/cricket.txt', encoding='utf-8')
load_dotenv()
docs = loader.load()
print(type(docs))

print(type(docs[0]))
print(docs[0].page_content)
print(docs[0].metadata)

template = PromptTemplate(template='Write 50 words summary of text \n {text}',
                          input_variables = ['text'])

parser = StrOutputParser()

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)

chain = template | chat_model | parser
response = chain.invoke({'text' : docs[0].page_content})
print(response)

