import os
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