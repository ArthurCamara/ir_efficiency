"""Run a time experiment"""

from model import CrossEncoderTrainer
from data_loaders import IRDatasetsLoader, InMemoryLoader, IndexedLoader


import os
import argparse
import logging
import accelerator as ac
from utils import get_free_gpus
from transformers import AutoTokenizer
import wandb

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# not used for ir_axioms
docs_path = "/ssd2/arthur/MsMarcoTREC/docs/msmarco-docs.tsv"
queries_path = "/ssd2/arthur/MsMarcoTREC/queries/msmarco-doctrain-queries.tsv"
qrels_path = "/ssd2/arthur/MsMarcoTREC/qrels/msmarco-doctrain-qrels.tsv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", type=str, choices=["ir_datasets", "indexed", "in_memory"], default="indexed")
    parser.add_argument("--parallel", type=str, choices=["accelerator", "DataParallel"], default="DataParallel")
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=1e-5)
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_per_gpu", type=int, default=8)
    args = parser.parse_args()
    used_gpus = get_free_gpus(args.n_gpus)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, max_length=512)
    if args.loader == "ir_datasets":
        train_dataset = IRDatasetsLoader(tokenizer, docs_path, queries_path, qrels_path)
    elif args.loader == "indexed":
        train_dataset = IndexedLoader(tokenizer, docs_path, queries_path, qrels_path)
    elif args.loader == "in_memory":
        train_dataset = InMemoryLoader(tokenizer, docs_path, queries_path, qrels_path)

    n_gpus = len(used_gpus)
    experiment_name = f"{args.loader}_{args.base_model}_{args.parallel}_{n_gpus}"

    accelerator = args.parallel.lower() == "accelerator"
    if not accelerator:
        args.batch_per_gpu = len(used_gpus) * args.batch_per_gpu
    if not accelerator or ac.Accelerator().is_main_process:
        wandb.init(
            project="ir_efficiency",
            entity="acamara",
            reinit=True,
            name=experiment_name,
            config=args,
        )

    trainer = CrossEncoderTrainer(experiment_name, args.base_model, accelerator)
    trainer.fit(train_dataset, n_steps=args.n_steps, train_batch_size=args.batch_per_gpu)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
