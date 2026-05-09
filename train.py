import argparse
import os
import joblib
import pandas as pd

from azureml.core import Run
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=100)

    args = parser.parse_args()

    run = Run.get_context()

    # Load dataset
    data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

    # Features and target
    x = data.drop("DEATH_EVENT", axis=1)
    y = data["DEATH_EVENT"]

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        class_weight='balanced'
    ).fit(x_train, y_train)

    # Accuracy
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))

    # AUC
    y_proba = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    run.log("AUC", float(auc))

    # Save model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()