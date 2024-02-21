import argparse
import os
import sys
from typing import List, Optional
import json

import datasets
from langchain.globals import set_debug
from langchain.callbacks import FileCallbackHandler
import fire

from gigachain_extensions.utils import run_chain
import gigachain_extensions.chains as chains
from eval_utils import compute_scores
from data_utils import write_results_to_log
from loguru import logger
from gigachain_extensions.pydantic_models import ASTEAnswer

def compute_metrics(sents, pred, true):
    sents = [
        sent.split() 
        for sent in sents
    ]
    pred = [ "; ".join(f'({", ".join(pred_i)})' for pred_i in pred_set) for pred_set in pred ]
    true = [ "; ".join(f'({", ".join(pred_i)})' for pred_i in pred_set) for pred_set in true ]
    return compute_scores(
        pred_seqs=pred,
        gold_seqs=true,
        sents=sents,
        io_format="extraction",
        task="aste"
    )


def get_chain(chain_str: str, dataset: datasets.Dataset = None, n_examples=40, mrr=None):
    chain_getter = getattr(chains, f"get_{chain_str}_chain")
    uses_full_dataset = "retrieve" in chain_str
    if uses_full_dataset:
        chain = chain_getter(dataset, n_examples=n_examples, mrr=mrr)
    else:
        examples = list(dataset.select(range(min(n_examples, len(dataset)))))
        for example in examples:
            example['triplets'] = ASTEAnswer(triplets=example['triplets'])
        chain = chain_getter(examples)
    return chain


def fix_preds_format(result_list: List[List[tuple]]):
    result_list = [
        [ triplet for triplet in triplets if all(el is not None for el in triplet) ]
        if triplets is not None else None
        for triplets in result_list
    ]
    n_broken = 0
    n_wrong_format = sum(el is None for el in result_list)
    print(f"n wrong format = {n_wrong_format}")
    result_list = [
        el if el is not None else []
        for el in result_list
    ]
    new_result_list = []
    for result in result_list:
        if len(result) == 3 and all(
            not isinstance(el, tuple) for el in result
        ):
            new_result_list.append([result])
        else:
            new_result = []
            for tup in result:
                if len(tup) == 3:
                    new_result.append(tup)
                else:
                    n_broken += 1
            new_result_list.append(new_result)
    print(f"n_broken generations: {n_broken}")
    return new_result_list


def main(
    *args,
    dataset_path: str,
    chain_str: str,
    train_subset = "train",
    eval_subset = "val",
    generate_only_for_col=None,
    debug=False,
    max_workers=2,
    n_examples=0,
    mrr: Optional[bool] = None,
    log_file_postfix: str = "",
):
    set_debug(debug)
    ds = datasets.load_from_disk(dataset_path)
    # ds[eval_subset] = ds[eval_subset].select(range(100))
    if debug:
        ds[eval_subset] = ds[eval_subset].select(range(5))
        max_workers = 1

    results_log_dir = './new_results_log'
    if not os.path.exists(results_log_dir):
        os.mkdir(results_log_dir)
    log_file_path = f"{results_log_dir}/{chain_str}-{dataset_path.split('/')[-1].split('.')[0]}-{eval_subset}{'-debug' if debug else ''}-{n_examples}shot-{log_file_postfix}.txt"
    result_dump_file_path = f"{results_log_dir}/{chain_str}-{dataset_path.split('/')[-1].split('.')[0]}-{eval_subset}{'-debug' if debug else ''}-{n_examples}shot-{log_file_postfix}.json"

    # does not print anything
    # logger.add(log_file_path, colorize=True, enqueue=True)
    # callbacks = [FileCallbackHandler(log_file_path)])
    callbacks = []
    # wordaround
    with open(log_file_path, 'a') as sys.stdout:
        print({
            "dataset_path": dataset_path,
            "chain_str": chain_str,
            "train_subset": train_subset,
            "eval_subset": eval_subset,
            "debug": debug,
            "max_workers": max_workers,
            "n_examples": n_examples,
        })
        chain = get_chain(chain_str, ds[train_subset], n_examples=n_examples, mrr=mrr)
        texts = ds[eval_subset]['text']
        if generate_only_for_col is not None:
            generate_only_for = ds[eval_subset][generate_only_for_col]
            texts = [ text for i, text in enumerate(texts) if generate_only_for[i] ]
            print(f"Will only be generating predictions for {len(texts)} out of {len(generate_only_for)}")
        result: List[ASTEAnswer] = run_chain(
            chain=chain, 
            texts=texts, 
            callbacks=callbacks,
            max_workers=max_workers,
            print_exceptions=True,
            batch_size=max_workers,
        )
        result: List[List[tuple]] = [
            [ (triplet.aspect_term, triplet.opinion_term, triplet.sentiment) for triplet in answer.triplets ]
            if answer is not None else None
            for answer in result
        ]
        result_fixed = fix_preds_format(result)
        if generate_only_for_col is not None:
            generate_only_for = ds[eval_subset][generate_only_for_col]
            new_result_fixed = []
            cnt = 0
            for i in range(len(generate_only_for)):
                if generate_only_for[i]:
                    new_result_fixed.append(result_fixed[cnt])
                    cnt += 1
                else:
                    new_result_fixed.append([])
            result_fixed = new_result_fixed
        with open(result_dump_file_path, "w") as f:
            json.dump(result_fixed, f, ensure_ascii=False)
        raw_scores, fixed_scores, _, _, _ =compute_metrics(
            ds[eval_subset]['text'],
            result_fixed, 
            ds[eval_subset]['triplets']
        )

    write_results_to_log(log_file_path, raw_scores)
    write_results_to_log(log_file_path, fixed_scores)


if __name__ == "__main__":
    fire.Fire(main)
