from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# sequential chaining

template1 = PromptTemplate(template="Generate Detailed Report on {topic}",
                          input_variables=['topic'])


template2 = PromptTemplate(template="Generate 3 point summary from the following text \n {text}",
                          input_variables=['text'])
parser = StrOutputParser()

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 150)
chat_model = ChatHuggingFace(llm=llm)

chain = template1 | chat_model | parser | template2 | chat_model | parser
response = chain.invoke({'topic', 'Digital India'})
print(response)
chain.get_graph().print_ascii()


