from transformers import AutoModelForMaskedLM, AutoTokenizer

model_checkpoint = "cl-tohoku/bert-base-japanese-whole-word-masking"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model.save_pretrained("../../model/")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.save_pretrained("../../model/")
