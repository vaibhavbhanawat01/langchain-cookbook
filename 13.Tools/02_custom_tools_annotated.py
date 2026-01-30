from langchain_core.tools import tool


# step 1 function creation

@tool
def multiply(a: int, b: int) -> int:
    """Multiple Two numbers"""  # this will help LLM to recognize method and also add type hint
    return a * b

response = multiply.invoke({'a': 3, 'b': 6})
print(response)

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())