import argparse
import os
import sys

import datasets
from langchain.globals import set_debug
from langchain.callbacks import FileCallbackHandler

from utils import run_chain
import chains
from eval_utils import compute_scores
from data_utils import write_results_to_log
from loguru import logger

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


def get_chain(chain_str: str, dataset: datasets.Dataset = None, dataset_path:str = None):
    chain_getter = getattr(chains, f"get_{chain_str}_chain")
    uses_full_dataset = "retrieve" in chain_str
    if uses_full_dataset:
        chain = chain_getter(dataset_path)
    else:
        examples = list(dataset.select(range(min(40, len(dataset)))))
        chain = chain_getter(examples)
    return chain


def set_triplets(entry):
    entry['triplets'] = entry['triplets_python_str']
    return entry


def fix_preds_format(result_list):
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
    dataset_path: str,
    chain_str: str,
    train_subset = "train",
    eval_subset = "val",
    debug=False,
    max_workers=10,
):
    set_debug(debug)
    ds = datasets.load_from_disk(dataset_path)
    if debug:
        ds[eval_subset] = ds[eval_subset].select(range(5))
        max_workers = 1

    results_log_dir = './results_log'
    if not os.path.exists(results_log_dir):
        os.mkdir(results_log_dir)
    log_file_path = f"{results_log_dir}/{chain_str}-{dataset_path.split('/')[-1].split('.')[0]}-{eval_subset}.txt"

    ds = ds.map(set_triplets)
    chain = get_chain(chain_str, ds[train_subset], dataset_path + f"/{train_subset}")
    # does not print anything
    # logger.add(log_file_path, colorize=True, enqueue=True)
    # callbacks = [FileCallbackHandler(log_file_path)])
    callbacks = []
    # wordaround
    with open(log_file_path, 'w') as sys.stdout:
        result = run_chain(
            chain=chain, 
            texts=ds[eval_subset]['text'], 
            callbacks=callbacks,
            max_workers=max_workers
        )
        result_fixed = fix_preds_format(result)
        raw_scores, fixed_scores, _, _, _ =compute_metrics(
            ds[eval_subset]['text'],
            result_fixed, 
            ds[eval_subset]['triplets']
        )

    write_results_to_log(log_file_path, raw_scores)
    write_results_to_log(log_file_path, fixed_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default='./data/bank_sentenized', type=str, required=False)
    parser.add_argument("--chain", type=str, required=True)
    parser.add_argument("--train-subset", default="train", type=str, required=False)
    parser.add_argument("--eval-subset", default="val", type=str, required=False)
    parser.add_argument("--max-workers", default=10, type=int, required=False)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(
        dataset_path=args.src, 
        chain_str=args.chain,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
        debug=args.debug,
        max_workers=args.max_workers,
    )

