from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
url = 'https://www.amazon.in/Nothing-Phone-Lite-Black-128/dp/B0G3X48FQ4?pf_rd_p=aa8f1aab-ca71-47ca-9612-e4f6c0d85911&pf_rd_r=71BPXNKSE0M03JVHAPR1&ref_=Smartphones-S3P_B0G3X48FQ4'
loader = WebBaseLoader(url)
# we can also give list of URL, this will return multiple docs for each URLs
docs = loader.load();
#print(docs[0].page_content)

template = PromptTemplate(template= 'Answer the following question {question} from text {text}',
                          input_variables = ['questions', 'text'])
parser = StrOutputParser()

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 100)
chat_model = ChatHuggingFace(llm=llm)

chain = template | chat_model | parser

response = chain.invoke({
    'question' : 'What is the product name ?',
    'text': docs[0].page_content
})
print(response)