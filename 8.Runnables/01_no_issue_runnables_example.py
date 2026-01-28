import random

class LLMDummy:
    def __init__(self):
        print('LLM created')

    def predict(self, prompt):
        response_list = ['Virat Kohli is batsman', 'Delhi is capital of India', 'AI Stands for artifical Intelligence']
        return random.choice(response_list)
llm = LLMDummy()


class PromptTemplateDummy:
    def __init__(self, template,input_variables):
        self.template = template
        self.input_variables = input_variables
    def format(self, input_dict):
        return self.template.format(**input_dict)
template = PromptTemplateDummy(template='write a poem about {topic}', input_variables=['topic'])
prompt = template.format({'topic': 'india'})
response = llm.predict(prompt)
#print(response)


class LLMChain:
    def __init__(self, template, llm):
        self.template = template
        self.llm = llm
    def run(self, input_dict):
        prompt = self.template.format(input_dict)
        return self.llm.predict(prompt)
    
llmChain = LLMChain(template=template, llm=llm)
response = llmChain.run({'topic': 'America'})
print(response)

# now if we need to add multiple component then we need to create more chains components.