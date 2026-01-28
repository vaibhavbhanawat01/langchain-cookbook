from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline(
    model_id = 'HuggingFaceH4/zephyr-7b-beta',
    task = 'text-generaton',
    pipeline_kwargs=dict(
        max_new_tokens=10,
        temperature=0.5,
        do_sample=False,
        repetition_penalty=1.03,
    )
)

chat_model  = ChatHuggingFace(llm = llm)
response = chat_model.invoke("What is capital of delhi")
print(response)