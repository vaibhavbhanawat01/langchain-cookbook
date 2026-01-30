from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MultipleInput(BaseModel):
    a: int = Field(description="The first number", required = True)
    b: int = Field(description="The Second number", required = True)

class MultiplyTool(BaseTool):
    name: str =  'Multiply'
    description: str =  'Multiply Two Numbers'
    args_schema: type[BaseModel] = MultipleInput

    def _run(self, a: int, b: int) -> int:
        return a*b
tool = MultiplyTool()
response = tool.invoke({'a': 6, 'b': 7})
print(response)
print(tool.name)
print(tool.description)
print(tool.args)