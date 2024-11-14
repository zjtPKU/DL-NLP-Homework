import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from dataHelper import get_dataset
from torch.optim import Adam

import numpy as np
import torch
import evaluate
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
    set_seed,
)
from transformers import RobertaModel, RobertaTokenizer
from adapters import RobertaAdapterModel
from adapters import AdapterConfig
from peft import PeftConfig

# 加载 RoBERTa 预训练模型
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="restaurant_sup", metadata={"help": "The name of the dataset to use"}
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "Maximum input sequence length after tokenization"}
    )
    exp_name: str = field(
        default="debug", metadata={"help": "Experiment name"}
    )
    few_shot: Optional[bool] = field(default=False, metadata={"help": "Whether to use few-shot mode"})
    few_shot_size: Optional[int] = field(default=None, metadata={"help": "Size for few-shot samples"})

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

datasets = ["restaurant_sup", "acl_sup", "agnews_sup"]
models = ["roberta-base"]
num_labels = [3,6,4]
for dataset in datasets:
    for model_name in models:
        
        # Dynamically load dataset and obtain `num_labels` for each dataset
        raw_datasets = get_dataset(dataset, sep_token="<SEP>")
        if dataset == "restaurant_sup":
            num_labels=3
        elif dataset == "acl_sup":
            num_labels=6
        else:
            num_labels=4
        # Set up model and data arguments
        model_args = ModelArguments(model_name_or_path=model_name)
        data_args = DataTrainingArguments(dataset_name=dataset, exp_name=f"{model_name}_{dataset}")

        # Setting up wandb for tracking
        wandb.init(
            project="transformer-training-task3",
            name=data_args.exp_name,
            reinit=True
        )

        # Setting training arguments
        training_args = TrainingArguments(
            evaluation_strategy="steps",
            eval_steps=200,
            output_dir=f"./output/{data_args.exp_name}",
            num_train_epochs=10,
            logging_dir=f"./logs/{data_args.exp_name}",
        )

        set_seed(training_args.seed)

        # Load model, tokenizer, and configuration with dynamically set num_labels
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
        from peft import LoraConfig

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model =  RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)
        model.add_adapter(peft_config,dataset)
        # model.set_active_adapters(dataset)
        
        
        # Tokenize the datasets
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=data_args.max_seq_length)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Load evaluation metrics
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
            micro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]
            macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
            return {"accuracy": accuracy, "micro_f1": micro_f1, "macro_f1": macro_f1}

        # Initialize Trainer

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train and evaluate the model
        trainer.train()
        trainer.evaluate()

        # Finish wandb run
        wandb.finish()
