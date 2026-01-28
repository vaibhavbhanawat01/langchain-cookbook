from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# data validation and parsing the data using pydantic 
class Student(BaseModel):
    name: str = 'vaibhav'
    age: Optional[int] = None
    email: EmailStr 
    cgpa: float = Field(ge=0, le=10, default = 9, description = 'A decimal value representing the cgpa of the student')
new_student= {"name": 'Deepak', "age": '40', 'email': 'abc@gmail.com', 'cgpa': 8}
student = Student(**new_student) # unpacking of dict
print(dict(student))

print(student.model_dump_json())
 