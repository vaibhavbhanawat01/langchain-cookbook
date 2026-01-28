from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)
# if we need to enforce schema then we use this parser. StructuredOutputParser has been deprecated and 
# removed from new version of langchain and also we cannot do data validation

schema = [
    ResponseSchema(name = 'fact1', description = 'Fact1 about the topic'),
    ResponseSchema(name = 'fact2', description = 'Fact2 about the topic'),
    ResponseSchema(name = 'fact3', description = 'Fact3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(template = "Give me 3 facts about {topic} \n {format_instructions}",
                          input_variables=['topic'],
                          partial_variables={'format_instructions': parser.get_format_instructions()})

# prompt = template.invoke({'topic', 'Sachin Tendulkar'})
# response = chat_model.invoke(prompt)
# print(parser.parse(response.content))

chain = template | chat_model | parser
response = chain.invoke({'topic', 'Virat Kohli'})
print(response)