import time
from os import PathLike
from typing import Iterable, Optional, Type, Union

import numpy as np
import psutil
import torch
import transformers
from accelerate import Accelerator
from accelerate.kwargs_handlers import DistributedDataParallelKwargs
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, PreTrainedModel
import wandb


class CrossEncoderModel(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.bert_model = AutoModel.from_pretrained(config._name_or_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None, return_logits=False):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        CLS_tokens = outputs.last_hidden_state[:, 0, :]
        pooled_outputs = self.dropout(CLS_tokens)
        logits = self.classifier(pooled_outputs).view(-1)
        if labels is not None:
            loss = self.loss(logits.view(-1), labels).mean()
        else:
            loss = 0.0  # meaningless
        if return_logits:
            return loss, logits
        return loss


class CrossEncoderTrainer:
    def __init__(
        self,
        experiment_name: str,
        model_name: Union[str, PathLike] = "distilbert-base-uncased",
        accelerator: bool = False,
        n_gpus: int = 2,
        use_wandb: bool = True,
    ) -> None:

        self.use_wandb = use_wandb
        self.is_main = True
        self.use_accelerator = accelerator

        if self.use_accelerator:
            kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
            self.accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
            if not self.accelerator.is_main_process:
                self.is_main = False

        self.name = experiment_name
        self.all_losses = []

        self.model_config = AutoConfig.from_pretrained(model_name)
        if self.use_accelerator:
            self.model = CrossEncoderModel(self.model_config)
            self.device = self.accelerator.device
        else:
            self.model = CrossEncoderModel(self.model_config)
            if n_gpus > 1:
                self.model = nn.DataParallel(self.model)
            self.device = torch.device("cuda")
            self.model.to(self.device)
        if self.use_wandb:
            wandb.watch(self.model)

    def fit(
        self,
        train_dataset: Iterable,
        weight_decay: int = 0.01,
        optimizer_class: Type[Optimizer] = transformers.AdamW,
        train_batch_size: int = 16,
        n_steps: int = 1000,
        lr: float = 2e-5,
        profiler: Optional[Type[torch.profiler.profile]] = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> None:
        sec_per_batch = []

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        train_loader.collate_fn = train_dataset.cross_encoder_batcher

        optimizer_params = {"lr": lr}
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if self.use_accelerator:
            self.model, optimizer, train_loader = self.accelerator.prepare(self.model, optimizer, train_loader)
        global_step = 0
        disable_tqdm = not self.is_main

        optimizer.zero_grad()
        self.model.train()
        number_of_samples = 0

        # train for a fixed ammount of steps, not epochs.

        pbar = tqdm(desc="Training", total=n_steps, ncols=90, disable=disable_tqdm)
        for features, labels in train_loader:
            step_start = time.perf_counter_ns()
            number_of_samples += len(labels)
            global_step += 1
            if global_step > n_steps:
                break
            pbar.update()
            if not self.use_accelerator:
                for k in features.keys():
                    features[k] = features[k].to(self.device)
                labels = labels.to(self.device)

            loss = self.model(**features, labels=labels)

            if self.use_accelerator:
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            else:
                loss = loss.mean(dim=0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if profiler is not None:
                profiler.step()
            time_elapsed = time.perf_counter_ns() - step_start
            if self.is_main and self.use_wandb:
                sec_per_batch.append(time_elapsed / 1e9)
                wandb.log({"loss": loss.item()})
                mem_used = psutil.Process().memory_info().rss // 1024**2
                wandb.log({"pid_mem": mem_used})
                wandb.log({"step_time": time_elapsed / 1e9})
        if self.is_main and self.use_wandb:
            wandb.run.summary["avg_time_per_batch"] = np.mean(sec_per_batch)
        if profiler is not None:
            profiler.stop()
            print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
