from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplication of Two numbers"""  # this will help LLM to recognize method and also add type hint
    return a * b
@tool
def addition(a: int, b: int) -> int:
    """Addition of Two numbers"""  # this will help LLM to recognize method and also add type hint
    return a + b

class MathamaticalOperationToolkit:
    def get_tools(self):
        return [multiply, addition]
toolkit = MathamaticalOperationToolkit();
for tool in toolkit.get_tools():
    print(f"toolname: {tool.name} description: {tool.description}")