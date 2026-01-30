from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultipleInput(BaseModel):
    a: int = Field(description="The first number", required = True)
    b: int = Field(description="The Second number", required = True)


def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name='multiply',
    description='Multiply Two Numbers',
    args_schema=MultipleInput
)

response = multiply_tool.invoke({'a':7, 'b': 10})

print(response)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)