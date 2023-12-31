import os
from tqdm.auto import tqdm
from pathlib import Path
from langchain.chat_models.gigachat import GigaChat
from langchain.vectorstores import SKLearnVectorStore

from loaders import HuggingFaceDatasetLoader
from embeddings import E5HuggingfaceEmbeddings


def parse_env_file(env_file_path):
    env_vars = {}
    with open(env_file_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            key, value = line.strip().split('=', 1)
            env_vars[key] = value 
    return env_vars


def setup_gigachat(env_file_path="./.env"):
    env = parse_env_file(env_file_path)
    chat = GigaChat(
        **env
    )
    return chat


    from tqdm.auto import tqdm

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_chain(texts, chain, max_workers=2, batch_size=32, print_exceptions=False):
    result = []
    batch_size = 32
    max_workers = 2
    for batch in tqdm(chunks(texts, batch_size), total=len(texts) // batch_size):
        try:
            result.extend(
                chain.batch(batch, config={"max_concurrency": max_workers})
            )
        except Exception as e:
            if print_exceptions:
                print(e)
            result.extend([ None ] * batch_size)
    return result


def get_retriever(dataset_path, k_examples=20):
    loader = HuggingFaceDatasetLoader(dataset_path, "text")
    data = loader.load()
    emb = E5HuggingfaceEmbeddings(
        model_name=os.path.expanduser("~") + "/models/multilingual-e5-small"
    )
    db = SKLearnVectorStore.from_documents(data, emb)
    retriever = db.as_retriever(search_kwargs={"k": k_examples})
    return retriever