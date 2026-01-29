from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(path='/Users/monika/Downloads/langchain-cookbook/9.DocumentLoaders/books'
                         , glob='*.pdf', loader_cls=PyPDFLoader)

# lazy loading lazy_load(), it load document on demand. Only required document it load on request
docs = loader.lazy_load()

for document in docs:
    print(document.metadata)