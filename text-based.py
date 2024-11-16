import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import PyPDF2
import openai

class PDFQA_System:
    def __init__(self):
        self.openai_api_key = "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" 
        openai.api_key = self.openai_api_key
        self.chroma_directory = '/Users/apple/Desktop/CapStone/chroma_db'
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.chroma_directory
        )

    def embed_text(self, text):
        embedding = self.embedding_model.embed(text)
        return embedding

    def add_pdf_to_collection(self, pdf_path):
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    embedding = self.embed_text(text)
                    self.vectorstore.add_documents(
                        documents=[{"content": text, "metadata": {"page_num": page_num}}],
                        embeddings=[embedding]
                    )
                    print(f"Added page {page_num + 1} to Chroma collection.")

    def generate_response(self, query):
        query_embedding = self.embed_text(query)
        results = self.vectorstore.query(query_embedding, top_k=3)
        context = " ".join([doc["content"] for doc in results])
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer concisely in less than 3 sentences."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        answer = response.choices[0].message['content'].strip()
        print("AI Response:", answer)

    def start_qa_loop(self):
        while True:
            query = input("Enter your question (or type 'stop' to end): ")
            if query.lower() == "stop":
                print("Ending session.")
                break
            self.generate_response(query)

pdf_path = "/Users/apple/Desktop/CapStone/03_linear_algebra - Copy.pdf"  # Replace with the actual path to your PDF
qa_system = PDFQA_System()
qa_system.add_pdf_to_collection(pdf_path)
qa_system.start_qa_loop()
