import os
import streamlit as st

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader



os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.pdf'


def process_doc(
        path: str = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf',
        is_local: bool = False,
        question: str = 'Cuál es el nombre del pdf?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)

    doc = loader.load_and_split()

    print(doc[-1])

    data = FAISS.from_documents(doc, embedding=OpenAIEmbeddings())
    pregunta= RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='map_reduce', retriever=data.as_retriever())
    print(pregunta.run(question))

    st.write(pregunta.run(question))

def resultado():
    st.title('PREGUNTAS DESDE IAUCE')
    uploader = st.file_uploader('Selecciona tu PDF', type='pdf')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('PDF subido!')

    question = st.text_input('Realizar una pregunta',
                             placeholder='Mandame una respuesta en base al documento pdf', disabled=not uploader)

    if st.button('HACER PREGUNTA'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('PDF Cargado')
            process_doc()

if __name__ == "__main__":
    resultado()


