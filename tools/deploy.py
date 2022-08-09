from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import time
import os

model_path = "../model/"
zipped_model_path = os.path.join(model_path, "model.tar.gz")
endpoint_name = "arps-bert-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

model = PyTorchModel(
    entry_point="inference.py",
    model_data=zipped_model_path,
    role=get_execution_role(),
    framework_version="1.11.0",
    py_version="py38",
)

predictor = model.deploy(
    initial_instance_count=1, instance_type="ml.m5.xlarge", endpoint_name=endpoint_name
)