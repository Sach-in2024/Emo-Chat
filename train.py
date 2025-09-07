# train.py (optimized for faster training)
import os
import random
import math
import numpy as np
import pandas as pd
from collections import Counter
from inspect import signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

# -------- Parameters --------
DATA_PATH = "data/train.txt"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/emotion_distilbert"
RANDOM_SEED = 42
NUM_EPOCHS = 1          # Reduced for fast testing
BATCH_SIZE = 32         # Increased for fewer steps
LR = 2e-5
MAX_LEN = 64            # Reduced sequence length
# ----------------------------

def read_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                text, label = line.rsplit(";", 1)
                text = text.strip()
                label = label.strip()
                if text == "" or label == "":
                    continue
                rows.append((text, label))
    return pd.DataFrame(rows, columns=["text", "label"])

def make_label_map(labels):
    uniq = sorted(list(set(labels)))
    label2id = {l:i for i,l in enumerate(uniq)}
    id2label = {i:l for l,i in label2id.items()}
    return label2id, id2label

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(model.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df = read_file(DATA_PATH)
    print("Loaded rows:", len(df))
    if len(df) == 0:
        raise SystemExit(f"No data found at {DATA_PATH} (expected 'text ; label' per line).")

    label2id, id2label = make_label_map(df["label"].tolist())
    df["label_id"] = df["label"].map(label2id)

    # Train/val/test split
    train_df, temp_df = train_test_split(df, stratify=df["label_id"], test_size=0.2, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, stratify=temp_df["label_id"], test_size=0.5, random_state=RANDOM_SEED)
    print("Split sizes:", len(train_df), len(val_df), len(test_df))

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    ds = ds.map(tokenize_batch, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])
    ds = ds.rename_column("label_id", "labels")

    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # Class weights (inverse frequency)
    counts = Counter(train_df["label_id"].tolist())
    weight_list = [0]*num_labels
    for label_id, cnt in counts.items():
        weight_list[label_id] = 1.0 / (cnt)
    weights = np.array(weight_list)
    weights = weights / weights.sum() * num_labels
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Version-robust TrainingArguments
    sig_params = signature(TrainingArguments.__init__).parameters
    supported_args = set(sig_params.keys())

    common_args = dict(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=False,    # disabled for faster training
    )

    extra_args = {}
    if "evaluation_strategy" in supported_args and "save_strategy" in supported_args:
        extra_args["evaluation_strategy"] = "epoch"
        extra_args["save_strategy"] = "epoch"
    else:
        steps_per_epoch = max(1, math.ceil(len(ds["train"]) / BATCH_SIZE))
        extra_args["eval_steps"] = steps_per_epoch
        extra_args["save_steps"] = steps_per_epoch
        if "evaluation_strategy" in supported_args:
            extra_args["evaluation_strategy"] = "steps"
        if "save_strategy" in supported_args:
            extra_args["save_strategy"] = "steps"

    training_args = TrainingArguments(**common_args, **extra_args)
    print("Training arguments keys used:", list(training_args.to_sanitized_dict().keys()))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Test set evaluation
    test_metrics = trainer.predict(ds["test"])
    print("Test metrics:", test_metrics.metrics)

    # Save label mapping
    import json
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
