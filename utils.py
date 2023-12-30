import os
from tqdm.auto import tqdm
from pathlib import Path
from langchain.chat_models.gigachat import GigaChat


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


def run_chain(texts, chain, max_workers=2, batch_size=32):
    result = []
    batch_size = 32
    max_workers = 2
    for batch in tqdm(chunks(texts, batch_size), total=len(texts) // batch_size):
        try:
            result.extend(
                chain.batch(batch, config={"max_concurrency": max_workers})
            )
        except Exception as e:
            result.extend([ None ] * batch_size)
    return result
