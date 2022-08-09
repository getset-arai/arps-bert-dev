import tarfile
import os

model_path = "../model/"
code_path = "../code/"

zipped_model_path = os.path.join(model_path, "model.tar.gz")

with tarfile.open(zipped_model_path, "w:gz") as tar:
    tar.add(model_path)
    tar.add(code_path)