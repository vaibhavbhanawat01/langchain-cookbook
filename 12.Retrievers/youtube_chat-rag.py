from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

video_id = 'Gfr50f6ZBvo'
try :
    transcripts_list = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=['en'])
    transcript = " ".join(chunk['text'] for chunk in transcripts_list)
    print(transcript)
except TranscriptsDisabled:
    print('No transcript available for this video')



print(transcripts_list)
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
chunks = splitter.create_documents([transcript])
print(len(chunks))
print(chunks[100])

embedding  = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


vector_store = FAISS.from_documents(chunks, embedding)

print(f" fetch all ids of documents {vector_store.index_to_docstore_id}")
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
retriever.invoke('What is deepmind')


llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 20)

chat_model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = llm.invoke(final_prompt)
print(answer.content)

# using chains 

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

response = main_chain.invoke('Can you summarize the video') 
print(response)