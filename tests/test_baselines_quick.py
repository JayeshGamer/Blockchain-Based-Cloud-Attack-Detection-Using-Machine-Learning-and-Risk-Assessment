import pandas as pd
from src.features.preprocessor import NetworkDataPreprocessor
from src.data.loader import split_data
from src.models.baselines import (
    DummyMajorityBaseline,
    LogisticRegressionBaseline,
    RandomForestBaseline,
    IsolationForestBaseline,
    StaticDSTBaseline,
)
from src.utils.metrics import compute_binary_metrics

def main():
    print("Loading test data...")
    df = pd.read_csv("data/processed/stratified_sample.csv.gz", nrows=5000)
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1)

    prep = NetworkDataPreprocessor()
    X_tr_df, y_tr, _, _ = prep.extract_and_clean_labels(train_df)
    X_te_df, y_te, _, _ = prep.extract_and_clean_labels(test_df)

    prep.fit(X_tr_df)
    X_tr = prep.transform(X_tr_df)
    X_te = prep.transform(X_te_df)

    baselines = [
        DummyMajorityBaseline(),
        LogisticRegressionBaseline(),
        RandomForestBaseline(n_estimators=30),
        StaticDSTBaseline(),
    ]

    for m in baselines:
        m.fit(X_tr, y_tr.values)
        preds = m.predict(X_te)
        probs = m.predict_proba(X_te)
        res = compute_binary_metrics(y_te.values, preds, probs)
        print(f"{m.name:25s} | Acc: {res['accuracy']:.4f} | F1: {res['macro_f1']:.4f} | Recall: {res['recall']:.4f} | FPR: {res['false_positive_rate']:.4f}")

    iso = IsolationForestBaseline(n_estimators=30)
    iso.fit(X_tr[y_tr.values == 0])
    preds = iso.predict(X_te)
    probs = iso.predict_proba(X_te)
    res = compute_binary_metrics(y_te.values, preds, probs)
    print(f"{iso.name:25s} | Acc: {res['accuracy']:.4f} | F1: {res['macro_f1']:.4f} | Recall: {res['recall']:.4f} | FPR: {res['false_positive_rate']:.4f}")

if __name__ == "__main__":
    main()
