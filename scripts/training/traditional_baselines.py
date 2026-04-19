import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TRAIN_PATH = "EA_train.csv"   # or "IA_train.csv"
TEST_PATH  = "EA_test.csv"    # or "IA_test.csv"
TARGET = "EA"              # <-- set to "EA" or "IA"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

train.columns = [c.strip() for c in train.columns]
test.columns  = [c.strip() for c in test.columns]

# --- checks ---
if "pat_query" not in train.columns or TARGET not in train.columns:
    raise ValueError(f"Train must contain 'pat_query' and '{TARGET}'. Got: {train.columns.tolist()}")

has_test_labels = TARGET in test.columns

# --- label mapping helpers ---
def map_binary(col):
    # For IA that uses 'Applicable'/'Not applicable' strings
    s = col.astype(str).str.strip().str.lower()
    mapped = s.map({"applicable": 1, "not applicable": 0})
    return mapped

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

def run_model(clf, name):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"\n=== {name} ({TARGET}) ===")
    if y_test is not None:
        print("Accuracy:", round(accuracy_score(y_test, pred), 4))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, digits=4))
    else:
        print("No test labels found — skipped metrics.")
        exit()

logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
svm = LinearSVC(class_weight="balanced")

_ = run_model(logreg, "logreg")
_ = run_model(svm, "svm")