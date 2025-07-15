from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# load ag dataset
dataset = load_dataset("ag_news")

# use DistilBert model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# tokenize text
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# apply preprocess to whole dataset
encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# define metrics
def compute_metrecs(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# training setup
training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    per_device_eval_batch_size = 16,
    per_device_train_batch_size = 16,
    num_train_epochs = 3,
    load_best_model_at_end = True,
    logging_dir = "./logs",
    logging_steps = 50
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset["test"],
    compute_metrics = compute_metrecs
)

# train model
trainer.train()

# Eval
predictions = trainer.predict(encoded_dataset["test"])
preds = np.argmax(predictions.predictions, axis=1)
print("\nClassification Report:\n")
print(classification_report(encoded_dataset["test"]["label"], preds, target_names=dataset["train"].features["label"].names))

