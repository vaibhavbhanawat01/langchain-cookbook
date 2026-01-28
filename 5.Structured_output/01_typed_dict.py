from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

# Typed Dict gives devloper hint about variable type but doesn't enforce to use specified type
personDict: Person = {'name': 'Vaibhav Bhanawat', 'age':33}
print(personDict)