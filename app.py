import streamlit as st
import os
import tempfile
from langchain.document_loaders import UnstructuredExcelLoader, JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def load_xlsx(uploaded_file):
    text = ""  # Initialize the text variable as an empty string
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    loader = UnstructuredExcelLoader(temp_file_path)
    docs_xls = loader.load()
    os.unlink(temp_file_path)  
    for page in docs_xls.pages:
        if page.extract_text():  # Ensure there's text to append
            text += page.extract_text()  # Concatenate text from each page
    return text



    #return docs_xls

def create_db_and_generate_answer(text):
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = db.as_retriever()

    #
    question = "You are an experienced sales person. Write a commercial offer in Russian that proposes the machinery matching the specifications the most. Include in your offer key data such as: capacity_kg,service_life_years,lifting_speed_mm_s,engine_type,lifting_height_mm,dimensions_mm."

    template = """Given the specifications {text}, answer the question using the following context to help: {context}.

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = OpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key)

    chain = (
        {"context": retriever, "question": RunnablePassthrough(), "text": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke({"question": question, "text": text})
    return response

def main():
    st.title("Big Tender")
    uploaded_file = st.file_uploader("Upload a file (.xlsx)", type='xlsx')
    if uploaded_file is not None:
        text = load_xlsx(uploaded_file)  #
        answer = create_db_and_generate_answer(text)  
        st.write("Generated Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
