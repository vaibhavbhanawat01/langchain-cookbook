from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

#dynamic prompt loading with json template and chaining the actions.

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 20)

chat_model = ChatHuggingFace(llm = llm)

papers = [
            "Attention is all You Need",
            "BERT: Pre-Training of Deep Bidirectional Transformers", 
            "GPT-3: Language Models are Few Shot Learners"
          ,"Diffusion Models Beat GANs on Image Synthesis"]
styles = ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
lengths = ["Short (1-2) paragraphs", "Medium (3-5) paragraphs", "Long (Detailed Explaination)"]

paper_selected = input(f"Select any paper from provided list. Select the Number (1-4) {papers} : ")
style_selected = input(f"Select any one of the style. Select the Number (1-4) {styles} : ")
length_selected = input(f"Select any one of the length of paragraph. Select the Number (1-3) {lengths} : ")


paper_input = papers[int(paper_selected) - 1]
style_input = styles[int(style_selected) - 1]
length_input = lengths[int(length_selected) - 1]

template = load_prompt("template.json")
chain = template | chat_model
response = chain.invoke({
    "length_input": length_input,
    "paper_input": paper_input,
    "style_input": style_input
})
print(response)