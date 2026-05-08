import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="Train student performance model")
parser.add_argument("--model", choices=["rf", "knn"], default="rf", 
                    help="Model type to save as model.pkl (rf=RandomForest, knn=KNeighbors)")
parser.add_argument("--n_neighbors", type=int, default=5,
                    help="Number of neighbors for KNN model")
args = parser.parse_args()

# Load dataset
data = pd.read_csv("student_data.csv")

# Features (Independent Variables)
X = data[["Attendance", "Assignment_Avg", "Study_Hours", "Mid_Marks"]]

# Target (Dependent Variable)
y = data["Final_Result"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest trained. Accuracy:", round(rf_acc * 100, 2), "%")

# KNN model pipeline (with scaling)
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=args.n_neighbors))
])
knn_pipeline.fit(X_train, y_train)
knn_pred = knn_pipeline.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print("KNN trained. Accuracy:", round(knn_acc * 100, 2), "%")

# Save both models
pickle.dump(rf_model, open("model_rf.pkl", "wb"))
pickle.dump(knn_pipeline, open("model_knn.pkl", "wb"))

# Set default model.pkl according to arg
if args.model == "knn":
    pickle.dump(knn_pipeline, open("model.pkl", "wb"))
    print("Default model set to KNN (model.pkl saved as KNN pipeline)")
else:
    pickle.dump(rf_model, open("model.pkl", "wb"))
    print("Default model set to RandomForest (model.pkl saved as RF model)")

print("Saved model_rf.pkl, model_knn.pkl, and model.pkl")