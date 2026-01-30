from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

#Built-in Tool - DuckDuckGo Search
duck = DuckDuckGoSearchRun()
response = duck.invoke('What top 5 current News on cricket sports?')
print(response)

print(duck.name)
print(duck.description)
print(duck.args)

#Built-in Tool - Shell Tool
# Execute shell commonds
shell = ShellTool();
response = shell.invoke('ls')

print(shell.name)
print(shell.description)
print(shell.args)

print(response)


