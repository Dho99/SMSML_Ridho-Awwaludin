import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Customer Churn Model")

# Memuat data hasil preprocessing
data = pd.read_csv("ecommerce_customer_churn_dataset_preprocessing.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Churned", axis=1),
    data["Churned"],
    random_state=42,
    test_size=0.2
)

param_distributions = {
    'n_estimators': [50, 100, 150, 200], 
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

n_iter_search = 5
best_accuracy = 0
input_example = X_train[0:5]


with mlflow.start_run(run_name="Random Search Optimization"):
    
    for i, params in enumerate(ParameterSampler(param_distributions, n_iter=n_iter_search, random_state=42)):
        
        with mlflow.start_run(run_name=f"Iteration_{i}", nested=True):
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], 
                min_samples_split=params['min_samples_split'], 
                random_state=42
            )
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)


            y_train_pred = model.predict(X_train)
            y_train_prob = model.predict_proba(X_train)[:, 1]

            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1]


            mlflow.log_metric("training_precision_score", precision_score(y_train, y_train_pred))
            mlflow.log_metric("training_recall_score", recall_score(y_train, y_train_pred))
            mlflow.log_metric("training_f1_score", f1_score(y_train, y_train_pred))
            mlflow.log_metric("training_accuracy_score", accuracy_score(y_train, y_train_pred))
            mlflow.log_metric("training_log_loss", log_loss(y_train, y_train_prob))
            mlflow.log_metric("training_roc_auc", roc_auc_score(y_train, y_train_prob))

            mlflow.log_metric("RandomForestClassifier_score_X_test", accuracy_score(y_test, y_test_pred))
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_test_pred))
           
           

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="best_model",
                    input_example=input_example,
                )
                mlflow.log_metric("best_accuracy", best_accuracy)

    print(f"Optimasi selesai. Akurasi terbaik: {best_accuracy}")