# --------------------------------------------------------------------------------
# 1) Install & Imports (If needed in a notebook environment)
# --------------------------------------------------------------------------------
# !pip install -U accelerate
# !pip install -U transformers
# !pip install datasets
# !pip install wandb

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import (
    RobertaTokenizer,
    RobertaModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
    logging
)
from datasets import Dataset, DatasetDict, load_metric
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------------------------------------
# 2) Basic Setup
# --------------------------------------------------------------------------------

# Set random seeds for reproducibility
set_seed(42)
np.random.seed(42)

# Suppress verbose logging from Transformers
logging.set_verbosity_error()

# Login to wandb (replace the key with your own or configure env variable)
wandb.login(key='')

# Select Device (CPU, CUDA, MPS if on Mac w/ M1)
device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using device: {device}")

# --------------------------------------------------------------------------------
# 3) Data Loading Function
# --------------------------------------------------------------------------------
def load_data(filename):
    """
    Loads data from a CSV file into a Dataset.
    Assumes there's a 'pat_query' column for text and 'EA' or 'emotional' for label.
    Renames 'EA' -> 'emotional' if needed.
    """
    df = pd.read_csv(filename)
    df['EA'] = df['EA'].map({'Not Applicable': 0, 'Applicable': 1})

    # If your CSV has a column 'EA' instead of 'emotional', rename:
    if 'EA' in df.columns and 'emotional' not in df.columns:
        df = df.rename(columns={'EA': 'emotional'})

    # Ensure the label is binary (0 or 1). If not, adjust accordingly.
    return Dataset.from_pandas(df)

# --------------------------------------------------------------------------------
# 4) Custom Model: RoBERTa + Attention Pooling
# --------------------------------------------------------------------------------
class RobertaWithAttentionPooling(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=2, hidden_size=768):
        super().__init__()
        # Load pretrained RoBERTa (encoder only)
        self.roberta = RobertaModel.from_pretrained(model_name)

        # A small feed-forward net to produce attention scores for each token
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

        # Classification layer for binary classification
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass inputs through RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Compute attention scores: [batch_size, seq_len, 1]
        attn_scores = self.attention(last_hidden_state).squeeze(-1)  # -> [batch_size, seq_len]

        # Mask out padding tokens with large negative number so they don't affect softmax
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax over seq_len to get attention weights: [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)

        # Weighted sum of hidden states to get a single vector [batch_size, hidden_size]
        pooled_output = torch.sum(last_hidden_state * attn_weights, dim=1)

        # Classifier
        logits = self.classifier(pooled_output)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# --------------------------------------------------------------------------------
# 5) Tokenizer and Data Preprocessing
# --------------------------------------------------------------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    """
    Tokenizes the 'pat_query' column.
    For binary classification, expects 'emotional' column as label (0 or 1).
    """
    # Tokenize text
    tokenized = tokenizer(
        examples['pat_query'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

    # Rename 'emotional' -> 'labels' for Trainer compatibility
    if 'emotional' in examples:
        tokenized['labels'] = [int(label) for label in examples['emotional']]
    return tokenized

# --------------------------------------------------------------------------------
# 6) Load and Prepare Datasets
# --------------------------------------------------------------------------------

# For Human version, load the EA_train.csv, 
# for Autonomous version, load the unseen_EA_IA_train_autonomous.csv
train_dataset = load_data('EA_train.csv')
val_dataset   = load_data('EA_eval.csv')
test_dataset  = load_data('EA_test.csv')

# Map tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

# Set correct format for PyTorch
columns = ["input_ids", "attention_mask", "labels"]
train_dataset.set_format(type="torch", columns=columns)
val_dataset.set_format(type="torch", columns=columns)
test_dataset.set_format(type="torch", columns=columns)

# --------------------------------------------------------------------------------
# 7) Initialize Model
# --------------------------------------------------------------------------------
model = RobertaWithAttentionPooling(
    model_name="roberta-base",
    num_labels=2,       # Binary classification
    hidden_size=768     # For roberta-base
).to(device)

# --------------------------------------------------------------------------------
# 8) Define the compute_metrics function
# --------------------------------------------------------------------------------
def compute_metrics(eval_pred: EvalPrediction):
    """
    Computes the accuracy and F1 score for classification.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    # Precision, Recall, F1
    precision = precision_score(labels, preds, average='macro')
    recall    = recall_score(labels, preds, average='macro')
    f1        = f1_score(labels, preds, average='macro')

    # # Specificity = TN / (TN + FP)
    # tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    # specificity = tn / (tn + fp)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
        # "specificity": specificity
    }

# --------------------------------------------------------------------------------
# 9) TrainingArguments & Trainer
# --------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results_project",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    num_train_epochs=10,                  # Adjust as needed
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    seed=42,
    report_to="wandb",                   # Enable W&B logging
    run_name="empathyEA"        # W&B run name
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# --------------------------------------------------------------------------------
# 10) Train the Model
# --------------------------------------------------------------------------------
trainer.train()

# --------------------------------------------------------------------------------
# 11) Evaluate on Test Set
# --------------------------------------------------------------------------------
test_results = trainer.evaluate(test_dataset)
print("Test Evaluation Results:", test_results)

# --------------------------------------------------------------------------------
# 12) Generate Predictions and Save to CSV
# --------------------------------------------------------------------------------
# Get predictions
predictions = trainer.predict(test_dataset)
pred_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()

# Load the original test dataset to get patient queries
test_df = pd.read_csv("EA_test.csv")
test_df["Predicted_Label"] = pred_labels

# Save to CSV
output_filename = "test_output_ea.csv"
test_df.to_csv(output_filename, index=False)

print(f"Predictions saved to {output_filename}")
# End of script