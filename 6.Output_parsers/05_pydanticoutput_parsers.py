from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()

#pydantic output parser

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description = "Name of the person")
    age: int = Field(ge = 18, description="Age of the person")
    city: str = Field(description= "City of the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(template = "Give me name, age, City of the fictional {place} person \n {format_instructions}",
    input_variables = ['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()})
# prompt = template.invoke({'place', 'Indian'})
# print(prompt)
# response = chat_model.invoke(prompt)
# print(parser.parse(response.content))

chain = template | chat_model | parser
response = chain.invoke({'place', 'Austraila'})
print(response)