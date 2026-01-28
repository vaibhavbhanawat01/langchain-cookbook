from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

#conditional chaining 

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 50)
chat_model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
      sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser = PydanticOutputParser(pydantic_object=Feedback)
text_parser = StrOutputParser()

template1 = PromptTemplate(template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}',
                          input_variables=['feedback'],
                          partial_variables={'format_instructions': parser.get_format_instructions()})

classifier_chain = template1 | chat_model | parser

#response = classifier_chain.invoke({'feedback': 'Product is terrible'});
#print(response.sentiment)

template2 = PromptTemplate(template='Write a appropriate response to this positive feedback \n {feedback}',
                           input_variables=['feedback'])
template3 = PromptTemplate(template='Write a appropriate response to this negative feedback \n {feedback}',
input_variables=['feedback'])

branch_chain = RunnableBranch(
      (lambda x: x.sentiment == 'positive', template2 | chat_model| text_parser),
      (lambda x: x.sentiment == 'negative', template3 | chat_model| text_parser),
      RunnableLambda(lambda x: 'could not found sentiment')
)

chain = classifier_chain | branch_chain
response = chain.invoke({'feedback': 'Product is wonderfull'});
print(response)
chain.get_graph().print_ascii()
