import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from pathlib import Path

from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


dataset = datasets.load_from_disk("data/hg/banks_sentenized_w_emp")

def get_label(entry):
    entry['label'] = int(len(entry['triplets']) != 0)
    return entry

dataset = dataset.map(get_label)

for i in range(30):
    print(dataset['train'][i])

model_path = Path(os.path.expanduser("~")) / "models" / "ruRoberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function)

train_ds = tokenized_dataset['train']
val_ds = tokenized_dataset['val']

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=200)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    roc_auc = roc_auc_score(y_true=labels, y_score=predictions[:, 1])
    f1 = f1_score(y_true=labels, y_pred=predictions.argmax(axis=-1))
    return {"roc_auc": roc_auc, "f1_score": f1}


training_args = TrainingArguments(
    output_dir="./bert_results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()





