import os
from typing import List

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

from prompts import (
    get_fewshot_gen_aspect_opinion_prompt, 
    get_fewshot_gen_aspect_prompt,
    get_fewshot_gen_polarity_from_aspects_opinions_prompt,
    get_fewshot_aop_prompt
)
from utils import setup_gigachat, get_retriever
from parsers import AnyListOutputParser
from example_selectors import ASTE_AO_RetrieverExampleSelector

llm = setup_gigachat()

def get_a_chain(examples: List):
    aspect_examples = [
        {
            "text": example["text"], 
            "aspects": str([triplet[0] for triplet in example["triplets"]]).replace("'", "\"")
        }
        for example in examples
    ]
    prompt = get_fewshot_gen_aspect_prompt(aspect_examples)
    chain = (
        {"text": RunnablePassthrough()}
        | prompt
        | llm
        | AnyListOutputParser()
    )
    return chain


def get_ao_chain(examples: List):
    duplet_examples = [
        {
            "text": example["text"], 
            "duplets": str([(triplet[0], triplet[1]) for triplet in example["triplets"]]).replace("'", "\"")
        }
        for example in examples
    ]
    prompt = get_fewshot_gen_aspect_opinion_prompt(duplet_examples)
    chain = (
        {"text": RunnablePassthrough(), "duplets": lambda x: []}
        | prompt
        | llm
        | AnyListOutputParser()
    )
    return chain


def get_ao_p_chain(examples: List):
    ao_chain = get_ao_chain(examples)
    examples = [
        {
            "text": example["text"], 
            "duplets": str([(triplet[0], triplet[1]) for triplet in example["triplets"]]).replace("'", "\""),
            "triplets": str([(triplet[0], triplet[1], triplet[2]) for triplet in example["triplets"]]).replace("'", "\"")
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
        | AnyListOutputParser()
    )
    return two_staged_chain


def get_aop_chain(examples: List):
    prompt = get_fewshot_aop_prompt(
        examples=examples
    )
    chain = (
        {"text": RunnablePassthrough()}
        | prompt
        | llm
        | AnyListOutputParser()
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
        | AnyListOutputParser()
    )
    return chain