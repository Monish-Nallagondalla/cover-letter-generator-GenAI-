import pandas as pd
import chromadb
import uuid
import docx


class Portfolio:
    def __init__(self, file_path="resource/my_resume.docx"):
        self.file_path = file_path
        self.data = self._read_word_file(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def _read_word_file(self, file_path):
        doc = docx.Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        # Assuming the resume is structured into sections
        # You might need to adjust this to extract Techstack and Links
        return pd.DataFrame([{'Techstack': '\n'.join(text), 'Links': ''}])

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])

