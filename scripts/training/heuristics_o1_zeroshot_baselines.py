import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

dim = 'EA'

try:
  # Load the human consensus test set on the dim
  file_path = dim+'_test.csv'
  df = pd.read_csv(file_path)
  # Map the True_Label column to create a new 'True_Label' column
  df['True_Label'] = df[dim].map({'Applicable': 1, 'Not Applicable': 0})
except:
  print("Load the EA_test.csv or IA_test.csv")
  exit()

try:
  # Load the o1 annotations on test set on the dim
  df_o1_baseline = pd.read_csv(f'o1_zeroshot_{dim}_test.csv')  # baseline output of o1 annotations on the test set
  # Map baseline predictions to binary
  df_o1_baseline['Baseline_Label'] = df_o1_baseline[f'{dim}_applicability'].map({'Applicable': 1, 'Not Applicable': 0})
except:
  print("Load the o1_zeroshot_EA_test.csv or o1_zeroshot_IA_test.csv")
  exit()


# Extract the true labels and the annotated labels by o1
true_labels = df['True_Label']
o1_predictions = df_o1_baseline['Baseline_Label']

# ------------------ o1 Performance ------------------
# Macro F1 and Weighted F1 for model predictions
o1_macro_f1 = f1_score(true_labels, o1_predictions, average='macro')
o1_weighted_f1 = f1_score(true_labels, o1_predictions, average='weighted')
o1_accuracy = accuracy_score(true_labels, o1_predictions)

# ------------------ Random Predictions ------------------
# Set random seed for reproducibility
np.random.seed(42)
random_predictions = np.random.choice([0, 1], size=len(df))

random_macro_f1 = f1_score(true_labels, random_predictions, average='macro')
random_weighted_f1 = f1_score(true_labels, random_predictions, average='weighted')
random_accuracy = accuracy_score(true_labels, random_predictions)


# ------------------ All Ones Predictions ------------------
all_ones_predictions = np.ones(len(df))

all_ones_macro_f1 = f1_score(true_labels, all_ones_predictions, average='macro')
all_ones_weighted_f1 = f1_score(true_labels, all_ones_predictions, average='weighted')
all_ones_accuracy = accuracy_score(true_labels, all_ones_predictions)


# ------------------ All Zeros Predictions ------------------
all_zeros_predictions = np.zeros(len(df))

all_zeros_macro_f1 = f1_score(true_labels, all_zeros_predictions, average='macro')
all_zeros_weighted_f1 = f1_score(true_labels, all_zeros_predictions, average='weighted')
all_zeros_accuracy = accuracy_score(true_labels, all_zeros_predictions)


# ------------------ Final Output ------------------
results = {
    "o1 Zero-shot Macro F1": o1_macro_f1,
    "o1_predictions Weighted F1": o1_weighted_f1,
    "o1_predictions Accuracy (%)": o1_accuracy,

    "Random Macro F1": random_macro_f1,
    "Random Weighted F1": random_weighted_f1,
    "Random Accuracy (%)": random_accuracy,

    "All Ones Macro F1": all_ones_macro_f1,
    "All Ones Weighted F1": all_ones_weighted_f1,
    "All Ones Accuracy (%)": all_ones_accuracy,

    "All Zeros Macro F1": all_zeros_macro_f1,
    "All Zeros Weighted F1": all_zeros_weighted_f1,
    "All Zeros Accuracy (%)": all_zeros_accuracy,
}

print(f"Baselines results for dimension {dim} : ")

# Print the results nicely
for k, v in results.items():
    print(f"{k}: {v:.4f}")