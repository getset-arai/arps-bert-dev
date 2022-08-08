import logging
from models.bert_finetuning import BertFinetuning
from utils.analyze_livedoor import AnalyzeLivedoor

logging.basicConfig(level=logging.INFO)

DIR_NAME = "./static/datasets/kaden-channel"
analyze_livedoor = AnalyzeLivedoor(DIR_NAME)

PRETRAINED_MODEL_PATH = "./static/model"
PRETRAINED_TOKENIZER_PATH = "./static/tokenizer"

finetuning = BertFinetuning(PRETRAINED_MODEL_PATH, PRETRAINED_TOKENIZER_PATH, analyze_livedoor.get_data_list())
result = finetuning.start_training()
if result:
    finetuning.save_weights("./static/weights")



