from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

markdown = '''
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 400,
    chunk_overlap=0
)

docs = splitter.split_text(markdown)
print(len(docs))
print(docs[0])
print(docs[1])