from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator='')
#chunk overlap to save context information while chunking text, or to minimize context loss
# due to splitting text in chunk.

# ratio of chunk size and chunk overlap is 20%, if 100 then overlap can be 10 - 20

# Direct text splitting
# text = '''Nature is the encompassing physical world and life, from subatomic particles to cosmic phenomena, providing essential resources like fresh air, water, and food, while also offering beauty, peace, and inspiration through its diverse landscapes, seasons, and creatures, though it faces threats from human activities like pollution and deforestation, making its preservation a critical duty for sustaining all life.
# Nature is the fundamental force that sustains life, encompassing all living and non-living things, from towering mountains and vast oceans to the smallest microbes and plants. It provides humans with necessities like oxygen, clean water, and food, while regulating vital processes such as decomposition and flood control. The beauty of nature, seen in sunrises, singing birds, and tranquil forests, offers rest and rejuvenation from daily stresses, highlighting the interconnectedness of ecosystems.'''
# response = splitter.split_text(text)
# print(response)
# print(len(response)) 


# load pdf then splitting text
loader = PyPDFLoader(file_path='/Users/monika/Downloads/langchain-cookbook/9.DocumentLoaders/dl-curriculum.pdf')
docs = loader.load()

result = splitter.split_documents(docs);
print(len(result))
print(result[0].page_content)
print(result[0].metadata)