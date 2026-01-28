from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)


# parser use to strict llm output to json, if cannot enforce json format keys so we use structure output parser.

parser = JsonOutputParser()

template = PromptTemplate(template = "Give me name, age, city of fictional person \n {format_instruction}",
                          input_variables=[],
                          partial_variables={
                              'format_instruction': parser.get_format_instructions()
                        })
# prompt = template.invoke({})
# response = chat_model.invoke(prompt)
# print(parser.parse(response.content)) 

chain = template | chat_model | parser
response = chain.invoke({})
print(response)
