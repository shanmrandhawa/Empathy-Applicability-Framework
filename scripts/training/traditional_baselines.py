import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

TRAIN_PATH = "EA_train.csv"   # or "IA_train.csv"
TEST_PATH  = "EA_test.csv"    # or "IA_test.csv"
TARGET = "EA"                 # <-- set to "EA" or "IA"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

train.columns = [c.strip() for c in train.columns]
test.columns  = [c.strip() for c in test.columns]

# --- checks ---
if "pat_query" not in train.columns or TARGET not in train.columns:
    raise ValueError(f"Train must contain 'pat_query' and '{TARGET}'. Got: {train.columns.tolist()}")

has_test_labels = TARGET in test.columns

# --- label mapping ---
def map_binary(col):
    s = col.astype(str).str.strip().str.lower()
    return s.map({"applicable": 1, "not applicable": 0})

y_train = map_binary(train[TARGET])

if has_test_labels:
    y_test = map_binary(test[TARGET])
else:
    print("No test labels found. Check the file to ensure EA or IA column is in it.")
    exit()

X_train_text = train["pat_query"].astype(str)
X_test_text  = test["pat_query"].astype(str)

# --- vectorize ---
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X_train = tfidf.fit_transform(X_train_text)
X_test  = tfidf.transform(X_test_text)

predictions = {}

def run_model(clf, name):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    predictions[name] = pred
    print(f"\n=== {name} ({TARGET}) ===")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=4))

run_model(LogisticRegression(max_iter=1000, class_weight="balanced"), "Logistic Regression")
run_model(LinearSVC(class_weight="balanced"), "Linear SVM")

# ------------------ McNemar vs. Transformer (Human Supervised) ------------------
def run_mcnemar(y_true, pred_model, pred_baseline, baseline_name):
    """Build contingency table and run McNemar test (transformer vs. one baseline)."""
    both_wrong    = ((pred_model != y_true) & (pred_baseline != y_true)).sum()
    only_baseline = ((pred_model != y_true) & (pred_baseline == y_true)).sum()
    only_model    = ((pred_model == y_true) & (pred_baseline != y_true)).sum()
    both_correct  = ((pred_model == y_true) & (pred_baseline == y_true)).sum()
    result = mcnemar([[both_wrong, only_baseline], [only_model, both_correct]], exact=False, correction=False)
    print(f"  Transformer vs {baseline_name}: p = {result.pvalue:.4f}")

try:
    df_human = pd.read_csv(f'output_human_{TARGET}_test.csv')
    df_merged = df_human[['pat_query', 'Predicted_Label']].merge(
        test[['pat_query', TARGET]], on='pat_query', how='inner'
    )
    y_true_aligned   = map_binary(df_merged[TARGET]).values
    transformer_preds = df_merged['Predicted_Label'].values

    print(f"\nMcNemar significance tests (Transformer Human Supervised vs. classical baselines) — {TARGET}:\n")
    for name, preds in predictions.items():
        # align baseline preds to merged index
        aligned_idx  = df_merged.index if len(preds) == len(df_merged) else range(len(df_merged))
        run_mcnemar(y_true_aligned, transformer_preds, preds[:len(df_merged)], name)

except FileNotFoundError:
    print(f"\n[McNemar skipped] 'output_human_{TARGET}_test.csv' not found. Run the RoBERTa classifier first.")