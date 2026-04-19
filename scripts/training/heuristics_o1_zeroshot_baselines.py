import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

dim = 'IA'  # Set to 'EA' or 'IA'

try:
    df = pd.read_csv(f'{dim}_test.csv')
    df['True_Label'] = df[dim].map({'Applicable': 1, 'Not Applicable': 0})
except:
    print("Load the EA_test.csv or IA_test.csv")
    exit()

try:
    df_o1_baseline = pd.read_csv(f'o1_zeroshot_{dim}_test.csv')
    df_o1_baseline['Baseline_Label'] = df_o1_baseline[f'{dim}_applicability'].map({'Applicable': 1, 'Not Applicable': 0})
except:
    print("Load the o1_zeroshot_EA_test.csv or o1_zeroshot_IA_test.csv")
    exit()

true_labels = df['True_Label']
o1_predictions = df_o1_baseline['Baseline_Label']

np.random.seed(42)
all_baselines = {
    "o1 Zero-Shot":          o1_predictions,
    "Random":                np.random.choice([0, 1], size=len(df)),
    "Always Applicable":     np.ones(len(df)),
    "Always Not Applicable": np.zeros(len(df)),
}

# ------------------ Helper functions ------------------
def print_metrics(name, y_true, y_pred):
    print(f"  {name}")
    print(f"    Accuracy:   {accuracy_score(y_true, y_pred):.4f}")
    print(f"    Macro-F1:   {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"    Wtd-F1:     {f1_score(y_true, y_pred, average='weighted'):.4f}")

def run_mcnemar(y_true, pred_model, pred_baseline, baseline_name):
    """Build contingency table and run McNemar test (model vs. one baseline)."""
    both_wrong    = ((pred_model != y_true) & (pred_baseline != y_true)).sum()
    only_baseline = ((pred_model != y_true) & (pred_baseline == y_true)).sum()
    only_model    = ((pred_model == y_true) & (pred_baseline != y_true)).sum()
    both_correct  = ((pred_model == y_true) & (pred_baseline == y_true)).sum()
    result = mcnemar([[both_wrong, only_baseline], [only_model, both_correct]], exact=False, correction=False)
    print(f"  Model vs {baseline_name}: p = {result.pvalue:.4f}")

def load_model_output(path):
    df_m = pd.read_csv(path)[['pat_query', 'Predicted_Label']]\
        .merge(df[['pat_query', 'True_Label']], on='pat_query', how='inner')\
        .merge(df_o1_baseline[['patient_query', f'{dim}_applicability']].rename(columns={'patient_query': 'pat_query'}), on='pat_query', how='inner')
    df_m['Baseline_Label'] = df_m[f'{dim}_applicability'].map({'Applicable': 1, 'Not Applicable': 0})
    return df_m

# ------------------ Baseline Metrics ------------------
print(f"\nBaseline results for dimension {dim}:\n")
for name, preds in all_baselines.items():
    print_metrics(name, true_labels, preds)

# ------------------ Model Metrics + McNemar (runs if model predictions are available) ------------------
try:
    df_human = load_model_output(f'output_human_{dim}_test.csv')
    human_preds    = df_human['Predicted_Label'].values
    y_true_aligned = df_human['True_Label'].values

    print(f"\nModel results for dimension {dim}:\n")
    print_metrics("Transformer (Human Supervised)", y_true_aligned, human_preds)

    try:
        df_auto = load_model_output(f'output_autonomous_{dim}_test.csv')
        print_metrics("Transformer (Autonomous-GPT Supervised)", df_auto['True_Label'].values, df_auto['Predicted_Label'].values)
    except FileNotFoundError:
        print(f"  [Autonomous Set skipped] 'output_autonomous_{dim}.csv' not found.")

    # McNemar: Human Set model vs. all baselines
    baseline_preds_aligned = {
        "o1 Zero-Shot":          df_human['Baseline_Label'].values,
        "Random":                np.random.choice([0, 1], size=len(df_human)),
        "Always Applicable":     np.ones(len(df_human)),
        "Always Not Applicable": np.zeros(len(df_human)),
    }

    print(f"\nMcNemar significance tests (Transformer-RoBERTa Human Supervised vs. baselines) — {dim}:\n")
    for name, preds in baseline_preds_aligned.items():
        run_mcnemar(y_true_aligned, human_preds, preds, name)

except FileNotFoundError:
    print(f"\n[Model metrics and McNemar skipped] 'output_human_{dim}.csv' not found. Run the RoBERTa classifier first.")