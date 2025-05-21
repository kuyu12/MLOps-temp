import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

model_name = "iris_classifier"
model_version = 1

mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2
)


model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

result = model.predict(X_test)

print(result)