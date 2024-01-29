from __future__ import annotations
import os
from typing import List, Any
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import asdict

from langchain.chat_models.gigachat import GigaChat
from langchain.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from datasets import Dataset

from gigachain_extensions.loaders import HuggingFaceDatasetLoader
from gigachain_extensions.embeddings import E5HuggingfaceEmbeddings


def parse_env_file(env_file_path):
    env_vars = {}
    with open(env_file_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            key, value = line.strip().split('=', 1)
            env_vars[key] = value 
    return env_vars


def setup_gigachat(env_file_path="./.env", model=None):
    env = parse_env_file(env_file_path)
    chat = GigaChat(
        **env
    )
    print(chat.to_json())
    return chat

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_chain(
    texts, 
    chain, 
    max_workers=2, 
    batch_size=32, 
    print_exceptions=False,
    callbacks=None
):
    if callbacks is None:
        callbacks = []
    result = []
    for batch in tqdm(chunks(texts, batch_size), total=len(texts) // batch_size):
        try:
            result.extend(
                chain.batch(batch, config={"max_concurrency": max_workers, "callbacks": callbacks}, verbose=True)
            )
        except Exception as e:
            if print_exceptions:
                print(e)
            result.extend([ None ] * len(batch))
    print(f"n broken entries: {sum(el is None for el in result)}. Rerun...")
    new_result = []
    for i, entry in tqdm(enumerate(result), total=len(result)):
        if entry is None:
            try:
                new_result.append(
                    chain.invoke(texts[i])
                )
            except Exception as e:
                new_result.append(None)
        else:
            new_result.append(entry)
    print(f"n broken entries: {sum(el is None for el in new_result)}.")
    return new_result


def get_retriever(dataset: List[dict[str, Any]], n_examples=10, content_col="text", triplets_col="triplets"):
    data: List[Document] = [
        Document(page_content=entry[content_col], metadata={"triplets": entry[triplets_col]})
        for entry in dataset
    ]
    emb = E5HuggingfaceEmbeddings(
        model_name=os.path.expanduser("~") + "/models/multilingual-e5-large"
    )
    db = SKLearnVectorStore.from_documents(data, emb)
    retriever = db.as_retriever(search_kwargs={"k": n_examples})
    return retriever