from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template = PromptTemplate(template="Give me 2 fact about {topic}",
                          input_variables=['topic'])

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 100)
chat_model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = template | chat_model | parser
response = chain.invoke({'topic', 'India'})
print(response)

chain.get_graph().print_ascii()

