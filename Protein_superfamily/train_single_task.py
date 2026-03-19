"""
train_single_task.py
--------------------
Trains a single task's classification head on a designated GPU.
Called by the notebook via subprocess.Popen.

"""

import os
import sys
import json
import math
import time
import random
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from collections import defaultdict

gpu_id = int(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    AutoConfig, RobertaPreTrainedModel, DebertaV2Model,
    TrainingArguments, default_data_collator, get_scheduler,
    DebertaTokenizerFast,
)
from transformers.trainer_pt_utils import get_parameter_names
from datasets import load_from_disk
from sklearn.metrics import root_mean_squared_error, roc_auc_score
from safetensors.torch import load_file
import bitsandbytes as bnb

logging.basicConfig(
    level=logging.INFO,
    format=f"[GPU {gpu_id}][%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class MLTClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.dropout  = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size // 2, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class RobertaMultiTaskModel(RobertaPreTrainedModel):
    def __init__(self, model_path1, model_path2, num_labels_list, problem_type_list):
        config = AutoConfig.from_pretrained(model_path1)
        super().__init__(config)
        self.config  = config
        self.config2 = AutoConfig.from_pretrained(model_path2)

        self.encoder1 = DebertaV2Model.from_pretrained(model_path1)
        self.encoder2 = DebertaV2Model.from_pretrained(model_path2)

        for param in self.encoder1.parameters(): param.requires_grad = False
        for layer in self.encoder1.encoder.layer[4:]:
            for param in layer.parameters(): param.requires_grad = True
        for param in self.encoder2.parameters(): param.requires_grad = False
        for layer in self.encoder2.encoder.layer[4:]:
            for param in layer.parameters(): param.requires_grad = True

        self.dense    = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dropout  = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        self.num_tasks   = len(num_labels_list)
        self.classification_heads = nn.ModuleList(
            [MLTClassificationHead(config, num_labels=n) for n in num_labels_list]
        )
        self.num_labels_list  = num_labels_list
        self.problem_type_list = problem_type_list
        self.post_init()

    def forward(
        self,
        input_ids=None, attention_mask=None,
        input_ids2=None, attention_mask2=None,
        input_ids3=None, attention_mask3=None,
        labels_list=None,
        output_attentions=None, output_hidden_states=None,
        return_dict=None, train=False, task_index=0,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        out1 = self.encoder1(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        p1 = out1.last_hidden_state[:, 0, :]

        if input_ids2 is not None and attention_mask2 is not None:
            p2 = self.encoder2(input_ids=input_ids2, attention_mask=attention_mask2,
                               return_dict=return_dict).last_hidden_state[:, 0, :]
            p3 = self.encoder2(input_ids=input_ids3, attention_mask=attention_mask3,
                               return_dict=return_dict).last_hidden_state[:, 0, :]
            combined = torch.cat((p1, p2, p3), dim=1)
        else:
            combined = torch.cat((p1, torch.zeros_like(p1), torch.zeros_like(p1)), dim=1)

        combined = self.dropout(combined)
        combined = self.dense(combined)
        combined = F.gelu(combined)
        logits   = self.classification_heads[task_index](combined)

        return {"logits": logits}


class MultiTaskWeightedTrainer:
    def __init__(self, model, datasets, ordered_tasks, optimizers=(None, None),
                 args=None, **kwargs):
        precision = "fp16" if args.fp16 else ("bf16" if args.bf16 else "no")
        self.accelerator = Accelerator(mixed_precision=precision)
        self.model     = self.accelerator.prepare(model)
        self.optimizer = self.accelerator.prepare(optimizers[0])

        self.batch_size                = args.per_device_train_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.data_collator             = kwargs.get("data_collator", default_data_collator)
        self.ordered_tasks             = ordered_tasks
        self.task_names                = list(ordered_tasks.keys())

        self.train_dataloaders  = self._prepare_dataloaders(datasets["train"], train=True)
        self.train_dataloaders2 = self._prepare_dataloaders(datasets["train"], train=False)
        self.eval_dataloaders   = self._prepare_dataloaders(datasets["test"],  train=False)

        total_micro_steps       = sum(len(dl) for dl in self.train_dataloaders.values())
        steps_per_epoch         = math.ceil(total_micro_steps / self.gradient_accumulation_steps)
        self.total_train_steps  = steps_per_epoch * args.num_train_epochs

        self.lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer,
            num_warmup_steps=self.total_train_steps // 10,
            num_training_steps=self.total_train_steps,
        )

        task_sizes              = {t: len(self.train_dataloaders[t]) for t in self.task_names}
        self.total_task_count   = sum(task_sizes.values())
        self.task_probabilities = np.array(list(task_sizes.values()), dtype=np.float32) / self.total_task_count

        log.info(f"Mixed precision: {self.accelerator.mixed_precision}")
        log.info(f"Task sizes: {task_sizes}")

    def _prepare_dataloaders(self, datasets, train=True):
        batch_multiplier = 1 if train else 2
        return {
            task: self.accelerator.prepare(DataLoader(
                dataset,
                batch_size=self.batch_size if (dataset.num_rows < 5000 and train)
                           else self.batch_size * batch_multiplier,
                shuffle=train,
                collate_fn=self.data_collator,
            ))
            for task, dataset in datasets.items()
        }

    def focal_bce_loss(self, logits, targets, weights=None, reduction="mean"):
        probs        = torch.sigmoid(logits)
        bce_loss     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_weight = ((1 - probs) ** 2) * targets + (probs ** 2) * (1 - targets)
        loss = focal_weight * bce_loss * (weights if weights is not None else 1.0)
        return loss.mean() if reduction == "mean" else loss.sum()

    def compute_loss(self, task_index, batch):
        labels    = batch.pop("labels")
        weights   = batch.pop("weights", None)
        task_name = batch.pop("task_name")
        logits    = self.model(**batch)["logits"]
        ptype     = self.ordered_tasks[task_name]["problem_type"]

        if ptype == "multi_label_classification":
            n = self.model.num_labels_list[task_index]
            return self.focal_bce_loss(logits.view(-1, n), labels.view(-1, n), weights)
        elif ptype == "regression":
            n = self.model.num_labels_list[task_index]
            return MSELoss()(logits.squeeze(), labels.squeeze() if n == 1 else labels)
        else:
            raise ValueError(f"Unknown problem type for task: {task_name}")

    def train_epoch(self):
        losses         = defaultdict(list)
        self.model.train()
        task_iters     = {t: iter(self.train_dataloaders[t]) for t in self.task_names}
        global_step    = 0
        max_prob       = max(self.task_probabilities)

        while any(v is not None for v in task_iters.values()):
            task_index = np.random.choice(len(self.task_names), p=self.task_probabilities)
            task       = self.task_names[task_index]
            if task_iters[task] is None:
                continue
            try:
                batch = next(task_iters[task])
                batch["task_name"]  = task
                batch["task_index"] = task_index

                scale = math.sqrt(max_prob / self.task_probabilities[task_index])
                loss  = self.compute_loss(task_index, batch)
                losses[task].append(loss.item())

                if global_step % 25 == 0:
                    log.info(f"Step {global_step}: {task}  loss={loss.item():.4f}")

                self.accelerator.backward(loss / self.gradient_accumulation_steps * scale)

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                global_step += 1
            except StopIteration:
                task_iters[task] = None

        if global_step % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return {t: float(np.mean(v)) if v else 0.0 for t, v in losses.items()}

    def evaluate_score(self, dataloader, dataset_type, metrics=None):
        if metrics is None:
            metrics = defaultdict(list)
        self.model.eval()

        for task_index, task_name in enumerate(self.task_names):
            predictions, true_labels, nan_indicators = [], [], []
            task_type = self.ordered_tasks[task_name]["problem_type"]

            for batch in dataloader[task_name]:
                labels      = batch.pop("labels")
                nan_ind     = batch.pop("weights", None)
                batch       = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                batch["task_index"] = task_index

                with torch.inference_mode():
                    logits = self.model(**batch)["logits"].detach()
                    preds  = logits if task_type == "regression" else torch.sigmoid(logits)

                predictions.append(preds.cpu().numpy())
                true_labels.append(labels.detach().cpu().numpy())
                if nan_ind is not None:
                    nan_indicators.append(nan_ind.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)
            if nan_indicators:
                nan_indicators = np.concatenate(nan_indicators, axis=0)

            if task_type == "regression":
                rmse    = root_mean_squared_error(true_labels, predictions)
                score   = {"rmse": rmse}
                average = rmse
            else:
                n_labels = self.ordered_tasks[task_name]["num_labels"]
                if n_labels > 1:
                    valid_aucs = []
                    for j in range(n_labels):
                        mask = nan_indicators[:, j] != 0
                        if len(np.unique(true_labels[mask, j])) > 1:
                            valid_aucs.append(float(roc_auc_score(true_labels[mask, j], predictions[mask, j])))
                        else:
                            valid_aucs.append(None)
                    score   = {"roc_auc": np.array(valid_aucs)}
                    average = np.nanmean([v for v in valid_aucs if v is not None])
                else:
                    auc     = roc_auc_score(true_labels, predictions, average=None)
                    score   = {"roc_auc": auc}
                    average = auc

            metrics[task_name].append(score)
            log.info(f"Eval [{dataset_type}] {task_name}: {average:.4f}  {score}")

        return metrics

    def train(self, num_epochs, eval_metrics=None, train_metrics=None, local_notebook_name=""):

        log.info("Epoch 0 (pre-training eval)")
        torch.cuda.empty_cache()
        eval_metrics = self.evaluate_score(self.eval_dataloaders, "eval", eval_metrics)
        for epoch in range(num_epochs):
            epoch_start = time.time()
            torch.cuda.empty_cache()
            log.info(f"Epoch {epoch + 1}/{num_epochs}")

            train_losses = self.train_epoch()
            epoch_mins   = (time.time() - epoch_start) / 60
            log.info(f"Epoch {epoch + 1} done in {epoch_mins:.1f} min | losses: {train_losses}")

            torch.cuda.empty_cache()

            if (epoch + 1) % 5 == 0:
                save_dir = f"my_trained_model_{local_notebook_name}/epoch_{epoch + 1}"
                os.makedirs(save_dir, exist_ok=True)
                self.model.save_pretrained(save_dir)
                torch.save({"epoch": epoch + 1, "optimizer": self.optimizer.state_dict()},
                           os.path.join(save_dir, "optimizer_state.pt"))
                torch.save(self.lr_scheduler.state_dict(),
                           os.path.join(save_dir, "scheduler_state.pt"))
                log.info(f"Checkpoint saved → {save_dir}")

            eval_metrics = self.evaluate_score(self.eval_dataloaders, "eval", eval_metrics)

        train_metrics = self.evaluate_score(self.train_dataloaders2, "train", train_metrics)
        return train_metrics, eval_metrics


def get_optimizer(model, n, retrain=False):
    training_args = TrainingArguments(
        output_dir=f"./models/multitask_model{n}",
        overwrite_output_dir=True,
        learning_rate=5e-5,
        weight_decay=5e-3,
        do_train=True,
        num_train_epochs=25 if retrain else 40,
        fp16=True,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
    )
    decay_params = get_parameter_names(model, [nn.LayerNorm])
    decay_params = [
        name for name in decay_params
        if "bias" not in name and model.get_parameter(name).requires_grad
    ]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if n in decay_params],
         "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n not in decay_params],
         "weight_decay": 0.0},
    ]
    optimizer = bnb.optim.AdamW8bit(
        grouped,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    return optimizer, training_args


def main():
    task_name     = sys.argv[1]
    # gpu_id already used at top of file
    save_dir_mlt  = sys.argv[3]
    notebook_name = sys.argv[4]
    retrain       = len(sys.argv) > 5 and sys.argv[5].lower() == "true"

    wall_start = time.time()
    log.info(f"Starting task: {task_name}")

    model_pth  = "../Pre-Training/PubchemModelDeberta"
    model_pth2 = "../Pre-Training/ProteinModelDeb"
    TRAIN_SAVE_PATH = "Data3/inputdataMMAT/train_datasets"
    TEST_SAVE_PATH  = "Data3/inputdataMMAT/test_datasets"
    TASKS_PATH      = "Data3/inputdataMMAT/tasks.json"

    log.info("Loading datasets from disk...")
    all_train = dict(load_from_disk(TRAIN_SAVE_PATH))
    all_test  = dict(load_from_disk(TEST_SAVE_PATH))
    with open(TASKS_PATH) as f:
        mmat_tasks = json.load(f)

    # Build ordered_tasks from mmat_tasks
    ordered_tasks = {
        k: {"problem_type": "regression", "num_labels": 1}
        for k in mmat_tasks
    }

    task_problem_type = ordered_tasks[task_name]["problem_type"]
    task_num_labels   = ordered_tasks[task_name]["num_labels"]

    single_task_datasets = {
        "train": {task_name: all_train[task_name]},
        "test":  {task_name: all_test[task_name]},
    }
    single_task_info = {task_name: {"problem_type": task_problem_type, "num_labels": task_num_labels}}
    local_notebook_name = notebook_name + task_name

    log.info("Building model...")
    model = RobertaMultiTaskModel(model_pth, model_pth2, [task_num_labels], [task_problem_type])
    
    if not retrain:
        saved_state_dict   = load_file(f"{save_dir_mlt}/model.safetensors")
        filtered_state_dict = {k: v for k, v in saved_state_dict.items()
                               if not k.startswith("classification_heads")}
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        log.info(f"Weights loaded. Missing (expected - heads): {len(missing)}  Unexpected: {len(unexpected)}")
    
        for param in model.parameters():            param.requires_grad = False
        for param in model.classification_heads.parameters(): param.requires_grad = True

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Frozen: {frozen:,}  Trainable: {trainable:,}")

    optimizer, training_args = get_optimizer(model, local_notebook_name, retrain)

    trainer = MultiTaskWeightedTrainer(
        model=model,
        datasets=single_task_datasets,
        ordered_tasks=single_task_info,
        args=training_args,
        optimizers=(optimizer, None),
    )

    train_metrics, eval_metrics = trainer.train(
        num_epochs=int(training_args.num_train_epochs),
        local_notebook_name=local_notebook_name,
    )

    results_dir = "task_results_SLT" if retrain else "task_results_tune"
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialisation
    def make_serialisable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict):       return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):       return [make_serialisable(v) for v in obj]
        if isinstance(obj, defaultdict): return make_serialisable(dict(obj))
        return obj

    torch.save(
        {"train": make_serialisable(train_metrics),
         "eval":  make_serialisable(eval_metrics)},
        f"{results_dir}/{task_name}_metrics.pt",
    )

    wall_mins = (time.time() - wall_start) / 60
    log.info(f" {task_name} complete in {wall_mins:.1f} min")


if __name__ == "__main__":
    main()
