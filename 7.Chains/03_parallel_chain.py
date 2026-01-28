from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

#Parallel chaining 

llm = HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',
                           task="text-generation",
                           max_new_tokens = 300)
chat_model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(template='Generate short and simple notes from the following text \n {text}',
                          input_variables=['text'])

template2 = PromptTemplate(template='Generate 3 short question and answer from the following text \n {text}',
                          input_variables=['text'])

template3 = PromptTemplate(template='Merge the provided notes and quiz into a single document \n Notes -> {notes} and quiz -> {quiz}',
                           input_variables=['notes', 'quiz'])
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': template1 | chat_model | parser,
    'quiz' : template2 | chat_model | parser
})

merged_chain = template3 | chat_model | parser

chain = parallel_chain | merged_chain
text = '''
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
'''
response = chain.invoke({
    'text' : text
})
print(response)
print(chain.get_graph().print_ascii())



