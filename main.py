"""Run a time experiment"""

from model import CrossEncoderTrainer


import os
import argparse
import logging
import torch
import accelerate as ac
from utils import get_free_gpus
from transformers import AutoTokenizer
from transformers import logging as t_logging
import wandb

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# stop complaining about unused imported weights
t_logging.set_verbosity_error()

# not used for ir_axioms
docs_path = os.path.join(os.environ["DATA_HOME"], "msmarco-docs.tsv")
queries_path = os.path.join(os.environ["DATA_HOME"], "msmarco-doctrain-queries.tsv")
qrels_path = os.path.join(os.environ["DATA_HOME"], "msmarco-doctrain-qrels.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loader",
        type=str,
        choices=["ir_datasets", "indexed", "in_memory"],
        default="ir_datasets",
        help="What type of document reader to use",
    )
    parser.add_argument(
        "--parallel",
        type=str,
        choices=["accelerator", "DataParallel"],
        default="DataParallel",
        help=(
            "What parallelism model to use. "
            "Accelerator uses DistributedDataParallel (multiple processes) "
            "while DataParallel uses one process with multiple trheads"
        ),
    )
    parser.add_argument("--n_gpus", type=int, default=2, help="number of GPUs to use")
    parser.add_argument("--n_steps", type=int, default=1000, help="total number of steps to run")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased", help="Base ranker miodel")
    parser.add_argument("--batch_per_gpu", type=int, default=8, help="batch size for each GPU")
    parser.add_argument("--profile", action="store_true", help="Run PyTorch Profiler with Tensorboard")

    # Extra parameters
    parser.add_argument("--pin_memory", action="store_true", help="use pin_memory=True on dataloader?")
    parser.add_argument("--num_workers", type=int, default=0, help="Use multiple workers on dataloader?")
    parser.add_argument("--ramdisk", action="store_true", help="transfer dataset to RAMDISK before loading?")

    args = parser.parse_args()
    used_gpus = get_free_gpus(args.n_gpus)

    accelerator = args.parallel.lower() == "accelerator"

    if not accelerator:
        args.batch_per_gpu = len(used_gpus) * args.batch_per_gpu
        n_gpus = len(used_gpus)

    else:
        n_gpus = ac.Accelerator().state.num_processes

    experiment_name = f"{args.loader}_{args.base_model}_{args.parallel}_{n_gpus}"
    prof = None
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, max_length=512)
    if args.ramdisk:
        os.environ["IR_DATASETS_HOME"] = "/dev/shm"
        experiment_name += "_ramdisk"
    if args.num_workers > 0:
        experiment_name += f"{args.num_workers}_workers"
    if args.pin_memory:
        experiment_name += "_pin_memory"

    # This is ugly, but enforces that ir_Datasets will read dataset from /dev/shm if needed
    from data_loaders import IRDatasetsLoader, InMemoryLoader, IndexedLoader

    # We are ignoring how long the startup-time of these are, in purpose, as these would only run once
    if args.loader == "ir_datasets":
        train_dataset = IRDatasetsLoader(tokenizer, docs_path, queries_path, qrels_path)
    elif args.loader == "indexed":
        train_dataset = IndexedLoader(tokenizer, docs_path, queries_path, qrels_path)
    elif args.loader == "in_memory":
        train_dataset = InMemoryLoader(tokenizer, docs_path, queries_path, qrels_path)
    if args.profile:
        print(
            "Running with profiler. "
            "Be aware that jobs with too many steps (i.e. too big json traces) can break TensorBoard!"
        )
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{experiment_name}"),
        )
        prof.start()

    use_wandb = False
    if (not accelerator or ac.Accelerator().is_main_process) and not args.profile:
        print(f"Running experiment with name {experiment_name} and logging to wandb.ai")
        use_wandb = True
        wandb.init(
            project="ir_efficiency",
            entity="acamara",
            reinit=True,
            name=experiment_name,
            config=args,
        )

    trainer = CrossEncoderTrainer(
        experiment_name,
        args.base_model,
        accelerator,
        n_gpus=args.n_gpus,
        use_wandb=use_wandb,
    )
    trainer.fit(
        train_dataset,
        n_steps=args.n_steps,
        train_batch_size=args.batch_per_gpu,
        profiler=prof,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
