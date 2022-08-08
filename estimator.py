import logging
from models.bert_proofreader import BertProofreader

logging.basicConfig(level=logging.INFO)

# PRETRAINED_MODEL = "cl-tohoku/bert-base-japanese-whole-word-masking"
PRETRAINED_MODEL_PATH = "./static/model"
PRETRAINED_TOKENIZER_PATH = "./static/tokenizer"
WEIGHTS_PATH = ""
proofreader = BertProofreader(PRETRAINED_MODEL_PATH, PRETRAINED_TOKENIZER_PATH)

# 気象庁は3日午前7時59分、青森県津軽地方で、線状降水帯による非常に激しい雨が同じ場所で降り続いているとして、「顕著な大雨に関する情報」を発表した。
# a1の性能をフルに使ってカメラ有効画素数約5010万画素・最高約30コマ/秒で撮影していると、その瞬間は以外にも早くやってきます。
# 1秒間に30コマというのは、ほとんど動画のようなものです。
result = proofreader.check_topk("a1の性能をフルに使ってカメラ有効画素数約5010万画素・最高約30コマ/秒で撮影していると、その瞬間は以外にも早くやってきます。", topk=100)
logging.info(f'result: {result}')
