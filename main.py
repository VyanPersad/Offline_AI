from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
    Reviews : {reviews}
    Question : {question}

"""
question = "What are the vegan options?"

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

reviews = retriever.invoke(question)

""""result = chain.invoke({
    "reviews":[],
    "question":"What are the main topics discussed in these reviews?"
})"""

result = chain.invoke({
    "reviews": reviews,
    "question": question
})

print(result)