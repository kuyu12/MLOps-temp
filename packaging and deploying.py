import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2
)

# Train model
max_depth = 1
model = RandomForestClassifier(n_estimators=10,max_depth=max_depth)
model.fit(X_train, y_train)

# Log model
with mlflow.start_run() as run:
    run_id = run.info.run_id
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="iris_classifier"
    )

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param('max_depth', max_depth)

# Register the model
# Can be manual or auto
model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(model_uri, "my_rf_classifier")

print(model_uri)
print(model_version)

# Using the model registry
# mlflow models serve -m "models:/my_rf_classifier/1" -p 5000

# Build Docker image
# mlflow models build-docker -m "models:/my_rf_classifier/1" -n "my-model-image"

# Run the container
# docker run -p 5000:8080 my-model-image