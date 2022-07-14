# `ir_efficiency`
This repository contains the code used in the experiments for our paper "Moving Stuff Around: A study on the efficiency of moving documents into memory for Neural IR models", published at the first [ReNeuIR workshop at the SIGIR 2022](https://reneuir.org).

You can find the paper [here](https://arxiv.org/pdf/2205.08343v2.pdf) and an open [Weights & Bias](https://wandb.ai/) dashboard with results [here](http://wandb.ai/acamara/ir_efficiency).

To re-run the experiments, first make sure you have CUDA installed on your machine (check [here](https://developer.nvidia.com/cuda-downloads) for instructions) and use the `Pipfile` to install the dependencies. We recommend using [Pipenv](https://pipenv.pypa.io/en/latest/) to do so in a new virtual environment. 

To run an experiment using DataParellel (i.e. multithreads), call the `main.py` file like this:

```bash
python main.py --loader ir_datasets --parallel DataParallel --n_gpus 8 \
               --n_steps 1000 --learning_rate 1e-5 --base_model distilbert-base-uncased\
               --batch_per_gpu 8 --pin_memory --num_workers 8 --ramdisk
```

For an experiment using `DistributedDataParallel` (i.e. using [Accelerate](https://github.com/huggingface/accelerate), use the `accelerate launch` command instead of `python`:
```bash
accelerate launch --config_file config_<n_gpus>.yaml main.py --loader ir_datasets —n_gpus <n_gpus> --parallel accelerator
 ```
Replacing `<n_gpus>` with how many GPUs you want to use. 

Other parameters are:

- `—loader` is the type of dataset loader to use. Options are `ir_datasets` `indexed` or `in_memory`
- `—parallel` is the parallelism strategy. Options are `accelerator` for using [Hugging Face's Accelerate](https://github.com/huggingface/accelerate) or `DataParallel` for the native `DataParallel` option.
-  `—n_gpus` is the number of GPUs to use in this experiment.
- `—n_steps`: Number of steps to train for
- `—learning_rate`: Learning rate for the optimiser
- `—base_model`: Base BERT model to use
- `--batch_per_gpu`: Number of GPUs to use for each experiment
- `—pin_memory`: wether or not to use the `pin_memory` option for PyTorch’s [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) object.
- `—num_workers` The number of workers (threads) to use when loading data from disk
- `—ramdisk`: Wether or not you want to use Ramdisk. If set to True, you must manually move the dataset to RAMDISK (usually, `/dev/shm`)

