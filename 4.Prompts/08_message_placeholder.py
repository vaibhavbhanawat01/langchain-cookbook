from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# use of append chat history as message placeholder

template = ChatPromptTemplate([
    ('system', 'You are helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('/Users/monika/Downloads/langchain-codebase/4.Prompts/chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = template.invoke({
    'query' : 'Where is my refund',
    'chat_history': chat_history
})

print(prompt)
