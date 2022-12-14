import logging
from lib.bert_finetuning import BertFinetuning
from analyze_livedoor import AnalyzeLivedoor

logging.basicConfig(level=logging.INFO)

DIR_NAME = "../../datasets/kaden-channel"
analyze_livedoor = AnalyzeLivedoor(DIR_NAME)

PRETRAINED_MODEL_PATH = "../../model/"

finetuning = BertFinetuning(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_PATH, analyze_livedoor.get_data_list())
result = finetuning.start_training()
if result:
    finetuning.save_weights("./static/weights")



