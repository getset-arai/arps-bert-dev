import json
import logging
import os
from lib.bert_proofreader import BertProofreader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JSON_CONTENT_TYPE = "application/json; charset=utf-8"


def model_fn(model_dir):
    logger.info(f'Call model_fn!: {model_dir}')
    model_path = os.path.join(model_dir, 'model/')
    return BertProofreader(model_path, model_path)


def input_fn(request_body, request_content_type=JSON_CONTENT_TYPE):
    logger.info("Call input_fn")
    if request_content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(request_body)
        logger.info(f'Json data: {input_data}')
        return input_data["text"]

    logger.error(f'Content_type invalid: {request_content_type}')
    input_data = {"errors": [f'content_type invalid: {request_content_type}']}
    return input_data


def predict_fn(input_data, model):
    logger.info("Call predict_fn")
    return model.check_topk(input_data)


def output_fn(prediction, content_type=JSON_CONTENT_TYPE):
    return json.dumps(prediction), content_type





