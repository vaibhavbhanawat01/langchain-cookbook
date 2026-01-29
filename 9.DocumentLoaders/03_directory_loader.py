from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(path='/Users/monika/Downloads/langchain-cookbook/9.DocumentLoaders/books'
                         , glob='*.pdf', loader_cls=PyPDFLoader)

# eager loading load(), it load all document in memory and also slow since we are loading all documents
docs = loader.load()

print(len(docs))

print(docs[0].metadata)
print(docs[0].page_content)