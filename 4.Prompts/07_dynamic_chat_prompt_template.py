from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# dynamic chat prompt change

template = ChatPromptTemplate([
    ('system', 'You are helpful {domain} expert'),
    ('human', 'Explain in simple terms, What is {topic}')
])

prompt = template.invoke({
    'domain': 'cricket',
    'topic': 'dusra'
})

print(prompt)