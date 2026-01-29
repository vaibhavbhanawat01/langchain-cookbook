from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path = '/Users/monika/Downloads/langchain-cookbook/9.DocumentLoaders/dl-curriculum.pdf')

# eager loading load()
docs = loader.load()

print(len(docs))

print(type(docs[0]))

print(docs[0].page_content)
print(docs[0].metadata)