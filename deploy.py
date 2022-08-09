from transformers import AutoModelForMaskedLM, AutoTokenizer
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import tarfile
import os
import time

model_path = "model/"
code_path = "code/"

# 事前学習済みモデルをダウンロード
model_checkpoint = "cl-tohoku/bert-base-japanese-whole-word-masking"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model.save_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.save_pretrained(model_path)

# パッケージ作成
zipped_model_path = os.path.join(model_path, "model.tar.gz")
with tarfile.open(zipped_model_path, "w:gz") as tar:
    tar.add(model_path)
    tar.add(code_path)

# デプロイ（コンテナ作成）
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

