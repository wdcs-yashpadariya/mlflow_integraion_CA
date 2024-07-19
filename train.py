import mlflow
import os
from ultralytics import YOLO

# Define paths for dataset and model
data_path = 'data.yaml'
model_path = 'yolov5n6u.pt'

# Initialize YOLO model
model = YOLO(model_path)

# Define training parameters
epochs = 3
batch_size = 32
img_size = 640

# Set MLflow tracking URI and create an experiment
mlflow.set_tracking_uri('http://127.0.0.1:5000')

# uncooment expereiment_id when u run script very first time in your system 

# experiment_id = mlflow.create_experiment(
#     name="Fire_detection",
#     artifact_location="Fire_detection_artifacts",
#     tags={"env": "dev", "version": "1.0.0"},
# )

mlflow.set_experiment(experiment_name="Fire_detection")

# Start MLflow run
with mlflow.start_run(run_name="Fire_2") as run:
    # Log parameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("img_size", img_size)

    # Train the model
    results = model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz=img_size)

    # Log metrics
    mlflow.log_metrics(results.metrics)

    # Log the model checkpoint to MLflow
    mlflow.pytorch.log_model(model, "yolo_model")

print("Training complete and logged to MLflow!")


