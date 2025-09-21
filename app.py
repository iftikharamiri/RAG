
import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate



#API_KEY = os.getenv("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")


# Sidebar contents
with st.sidebar:
    st.title('Allergi meny')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    stramlit, LangChain, and OpenAI.
    ''')
    add_vertical_space(5)
    st.write('powered by iftikhar')


load_dotenv()  # Load environment variables from .env file

def main():
    st.title("Allergi Chatbot")
    

    #upload PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf is not None:
        st.write(pdf.name)


    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)

        txt = ""
        for page in pdf_reader.pages:
            txt += page.extract_text()
        #st.write(txt)
       

        #split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(txt)
        #st.write(chunks)
        store_name = pdf.name[:-4]

        index_dir  = f"{store_name}_faiss"         # folder for the index

        if os.path.exists(index_dir):              # load if it already exists
            embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
            VectorStore = FAISS.load_local(
                index_dir, embeddings, allow_dangerous_deserialization=True
            )
            st.write("Embeddings loaded from disk")
        else:                                      # build and save
            embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(index_dir)
            st.write("Embeddings created and saved")

        #accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo", max_retries=3)

            nor_prompt = PromptTemplate(
                template=(
                    "Du er en hjelpsom assistent. Svar ALLTID på norsk (bokmål). "
                    "Bruk bare informasjonen i dokumentene. Hvis noe er uklart, si det tydelig.\n\n"
                    "{context}\n\nSpørsmål: {question}\nSvar:"
                ),
                input_variables=["context", "question"],
            )

            
            chain = load_qa_chain(llm, chain_type="stuff", prompt=nor_prompt)

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            st.write(response)