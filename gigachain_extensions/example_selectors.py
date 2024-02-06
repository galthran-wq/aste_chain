from langchain_core.example_selectors.base import BaseExampleSelector

from .pydantic_models import ASTEAnswer


class RetrieverExampleSelector(BaseExampleSelector):
    def __init__(self, retriever):
        self.retriever = retriever

    def add_example(self, example):
        raise NotImplementedError
    
    def format_docs(self, docs):
        return docs

    def select_examples(self, input_variables):
        # This assumes knowledge that part of the input will be a 'text' key
        new_text = input_variables["text"]
        best_k_docs = self.retriever.invoke(new_text)
        return self.format_docs(best_k_docs)


class ASTE_AO_RetrieverExampleSelector(RetrieverExampleSelector):
    def format_docs(self, docs):
        return [
            {
                "text": doc.page_content, 
                "duplets": ASTEAnswer(triplets=doc.metadata["triplets"]).model_dump_duplet_json()
            }
            for doc in docs
        ]


class AOP_RetrieverExampleSelector(RetrieverExampleSelector):
    def format_docs(self, docs):
        return [
            {
                "text": doc.page_content, 
                "triplets": ASTEAnswer(triplets=doc.metadata["triplets"]).model_dump_json()
            }
            for doc in docs
        ]