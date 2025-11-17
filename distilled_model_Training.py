from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from utilities import retrieve_dataset

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()

    labels = labels.astype(int)

    # ---- RELAXED ACCURACY: at least one matching label ----
    intersection = (preds & labels).sum(axis=1)

    # If intersection >= 1 → relaxed correct
    relaxed_correct = (intersection > 0).mean()

    return {
        "subset_accuracy": accuracy_score(labels, preds),
        "relaxed_accuracy": relaxed_correct,
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "samples_f1": f1_score(labels, preds, average="samples", zero_division=0),
    }

# Load Dataset
print("Loading dataset from utilities...")
dataset_dict = retrieve_dataset()

train_ds = dataset_dict["train"]
val_ds = dataset_dict["validation"]
test_ds = dataset_dict["test"]

print("Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))

# Tokenizer
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=192,
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Convert label list → multi-hot vector
NUM_LABELS = 28  # GoEmotions has 27 + neutral = 28

def pack_labels(example):
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    for label_id in example["labels"]:
        vec[label_id] = 1.0
    example["labels"] = vec
    return example


train_ds = train_ds.map(pack_labels)
val_ds = val_ds.map(pack_labels)
test_ds = test_ds.map(pack_labels)

# Remove unused columns
train_ds = train_ds.remove_columns(["text", "id"])
val_ds = val_ds.remove_columns(["text", "id"])
test_ds = test_ds.remove_columns(["text", "id"])

# Torch formatting
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load Model (correct classifier)
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)

# Custom collator to keep labels float32
base_collator = DataCollatorWithPadding(tokenizer)


def float_label_collator(batch):
    # convert labels to float32
    labels = torch.stack([item["labels"].float() for item in batch])

    # pad the other inputs
    items = [{k: v for k, v in item.items() if k != "labels"} for item in batch]
    padded = base_collator(items)

    padded["labels"] = labels
    return padded


# Training Arguments
training_args = TrainingArguments(
    output_dir="./go_emotions_model",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    num_train_epochs=6,
    weight_decay=0.01,
    fp16=True,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="no",
    remove_unused_columns=False,
    eval_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=float_label_collator,
)

# Train
print("Training starting...")
trainer.train()
print("Training finished!")

# Test Evaluation
print("Evaluating on test set...")
results = trainer.evaluate(test_ds)
print("Evaluation results:", results)
