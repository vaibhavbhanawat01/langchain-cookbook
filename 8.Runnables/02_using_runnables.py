import random

from abc import ABC, abstractmethod

class RunnableDummy(ABC):
    def invoke(self, input_dict):
        pass

class LLMDummy(RunnableDummy):
    def __init__(self):
        print('LLM created')
    def invoke(self, input_dict):
        response_list = ['Virat Kohli is batsman', 'Delhi is capital of India', 'AI Stands for artifical Intelligence']
        result = random.choice(response_list)
        return {'response': result}

    def predict(self, prompt):
        response_list = ['Virat Kohli is batsman', 'Delhi is capital of India', 'AI Stands for artifical Intelligence']
        result = random.choice(response_list)
        return {'response': result}
llm = LLMDummy()


class PromptTemplateDummy(RunnableDummy):
    def __init__(self, template,input_variables):
        self.template = template
        self.input_variables = input_variables
    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    
    def format(self, input_dict):
        return self.template.format(**input_dict)
# template = PromptTemplateDummy(template='write a poem about {topic}', input_variables=['topic'])
# prompt = template.format({'topic': 'india'})
# response = llm.predict(prompt)
# #print(response)
class RunnableConnector:
    def __init__(self, list_runnables):
        self.runnables = list_runnables
    def invoke(self, input_data):
        for runnable in self.runnables:
            input_data = runnable.invoke(input_data)
        return input_data
class StrOutputParserDummy():
    def __init__(self):
        pass
    def invoke(self, input_dict):
        return input_dict['response']
template = PromptTemplateDummy(template='write a poem about {topic}', input_variables=['topic'])
parser = StrOutputParserDummy()
connector = RunnableConnector([template, llm, parser])
response = connector.invoke({'topic' : 'Indian'})
print(response)