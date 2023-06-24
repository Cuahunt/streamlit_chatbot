
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback







def scrape_url(url):
    visited_urls = set()
    url_list = []

    def helper(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                for link in soup.find_all("a"):
                    href = link.get("href")
                    if href and href.startswith(url) and href not in visited_urls:
                        visited_urls.add(href)
                        url_list.append(href)
                        helper(href)
        except requests.RequestException as e:
            print(f"Failed to scrape URL '{url}': {str(e)}")

    helper(url)

    all_text = ""
    for url in url_list:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text(separator=" ")
                all_text += text
        except requests.RequestException as e:
            print(f"Failed to scrape URL '{url}': {str(e)}")

    return all_text



# pip install faiss,tiktoken, unstructured, libmagic(brew)
knowledge_base = None


def main():
    load_dotenv()
    st.set_page_config(page_title="I am your Chatbot")
    st.header("Ask me anything about your data")

    choice = st.selectbox(
        "Where do you want to get the data from", ["", "Website", "PDF"]
    )

    data = ""
    entered = False

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200
    )
    embeddings = OpenAIEmbeddings()

    if choice != "":
        if choice == "Website":
            url = st.text_input("Enter a URL to scrape")
            try:
                if url:
                    data = scrape_url(url)
                    docs = text_splitter.split_text(data)
                    knowledge_base = FAISS.from_texts(docs, embeddings)
                    entered = True
            except IndexError:
                st.error(
                    """
                    The website you entered cannot be processed.
                    Reenter a new URL.
                    """
                )

        elif choice == "PDF":
            pdf = st.file_uploader("Upload your PDF", type="pdf")
            if pdf is not None:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                data = text
                docs = text_splitter.split_text(data)
                knowledge_base = FAISS.from_texts(docs, embeddings)
                entered = True

        if entered:
            st.write("Extraction complete")
            if choice == "Website":
                user_question = st.text_input("Ask a question about the website:")
            elif choice == "PDF":
                user_question = st.text_input("Ask a question about the PDF:")

            if user_question:
                llm = OpenAI(temperature=0, verbose=True)
                docs = knowledge_base.similarity_search(user_question)
                chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
                st.write(response)


if __name__ == "__main__":
    main()
