from __future__ import annotations
import os
from typing import List, Literal, Union

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from datasets import Dataset

from prompts import (
    get_fewshot_gen_aspect_opinion_prompt, 
    get_fewshot_gen_aspect_prompt,
    get_fewshot_gen_polarity_from_aspects_opinions_prompt,
    get_fewshot_aop_prompt
)
from utils import setup_gigachat, get_retriever
from parsers import AnyListOutputParser
from example_selectors import (
    ASTE_AO_RetrieverExampleSelector,
    AOP_RetrieverExampleSelector,
)
from pydantic_models import ASTEAnswer

llm = setup_gigachat()

def get_a_chain(examples: List[dict[Literal['text'] | Literal['triplets'], str | ASTEAnswer]]):
    aspect_examples = [
        {
            "text": example["text"], 
            "aspects": example["triplets"].model_dump_aspect_json()
        }
        for example in examples
    ]
    prompt = get_fewshot_gen_aspect_prompt(aspect_examples)
    chain = (
        {"text": RunnablePassthrough()}
        | prompt
        | llm
        | PydanticOutputParser(pydantic_object=ASTEAnswer)
    )
    return chain


def get_ao_chain(examples: List[dict[Literal['text'] | Literal['triplets'], str | ASTEAnswer]]):
    duplet_examples = [
        {
            "text": example["text"], 
            "duplets": example["triplets"].model_dump_duplet_json(),
        }
        for example in examples
    ]
    prompt = get_fewshot_gen_aspect_opinion_prompt(duplet_examples)
    chain = (
        {"text": RunnablePassthrough(), "duplets": lambda x: []}
        | prompt
        | llm
        | PydanticOutputParser(pydantic_object=ASTEAnswer)
    )
    return chain


def get_ao_p_chain(examples: List[dict[Literal['text'] | Literal['triplets']]]):
    ao_chain = get_ao_chain(examples)
    examples = [
        {
            "text": example["text"], 
            "duplets": example["triplets"].model_dump_duplet_json(),
            "triplets": example["triplets"].model_dump_json(),
        }
        for example in examples
    ]
    prompt = get_fewshot_gen_polarity_from_aspects_opinions_prompt(
        examples=examples
    )
    two_staged_chain = (
        {"text": RunnablePassthrough(), "duplets": ao_chain}
        | prompt
        | llm
        | PydanticOutputParser(pydantic_object=ASTEAnswer)
    )
    return two_staged_chain


def get_aop_chain(examples: List[dict[Literal['text'] | Literal['triplets']]]):
    examples = [
        {
            "text": example["text"], 
            "triplets": example["triplets"].model_dump_json(),
        }
        for example in examples
    ]
    prompt = get_fewshot_aop_prompt(
        examples=examples
    )
    chain = (
        {"text": RunnablePassthrough(), "triplets": lambda x: []}
        | prompt
        | llm
        | PydanticOutputParser(pydantic_object=ASTEAnswer)
    )
    return chain


def get_retrieve_ao_chain(dataset_path: str, k_examples=20):
    retriever = get_retriever(dataset_path=dataset_path, k_examples=k_examples)
    example_selector = ASTE_AO_RetrieverExampleSelector(retriever)
    prompt = get_fewshot_gen_aspect_opinion_prompt(example_selector=example_selector)
    chain = (
        {"text": RunnablePassthrough(), "duplets": lambda x: [] } # duplets are populated by example selector
        | prompt
        | llm
        | PydanticOutputParser(pydantic_object=ASTEAnswer)
    )
    return chain


def get_retrieve_aop_chain(dataset: Dataset, k_examples=20):
    retriever = get_retriever(dataset=dataset, k_examples=k_examples)
    example_selector = AOP_RetrieverExampleSelector(retriever)
    prompt = get_fewshot_aop_prompt(example_selector=example_selector)
    chain = (
        {"text": RunnablePassthrough(), "triplets": lambda x: [] } # triplets are populated by example selector
        | prompt
        | llm
        | PydanticOutputParser(pydantic_object=ASTEAnswer)
    )
    return chain
