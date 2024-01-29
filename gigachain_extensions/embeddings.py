from typing import List
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class E5HuggingfaceEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [ f"passsage: {text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, query: str, passage=True) -> List[float]:
        query = f"{'query:' if not passage else 'passage:'} {query}"
        return super().embed_query(query)