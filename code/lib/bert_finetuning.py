import collections
import logging
import random

import numpy as np
from datasets import DatasetDict, Dataset
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM, AutoConfig,create_optimizer
from transformers.data import default_data_collator
import tensorflow as tf

logger = logging.getLogger(__name__)


def group_texts(examples):
    # Max 512
    chunk_size = 128
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


class BertFinetuning:
    def __init__(self, pretrained_model_path: str, pretrained_tokenizer_path: str, data_list: list):

        config = AutoConfig.from_pretrained(pretrained_model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path, config=config, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model_path, local_files_only=True)

        # データシャッフル トレーニングデータ:テストデータ = 9:1
        random.shuffle(data_list)
        n_train = int(0.9 * len(data_list))
        train_dataset = data_list[:n_train]
        test_dataset = data_list[n_train:]
        self.dataset = DatasetDict({
            "train": Dataset.from_dict({"text": train_dataset}),
            "test": Dataset.from_dict({"text": test_dataset})
        })

    def tokenize_function(self, examples):
        result = self.tokenizer(examples["text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    def start_training(self) -> bool:
        # トークン化
        tokenized_dataset = self.dataset.map(
            self.tokenize_function, batched=True, remove_columns=["text"]
        )
        # トークン調整(チャンクサイズ統一 labels生成)
        self.dataset = tokenized_dataset.map(group_texts, batched=True)
        logger.info(self.dataset)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        tf_train_dataset = self.dataset["train"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=data_collator,
            shuffle=True,
            batch_size=32,
        )
        tf_eval_dataset = self.dataset["test"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=data_collator,
            shuffle=False,
            batch_size=32,
        )
        num_train_steps = len(tf_train_dataset)
        optimizer, schedule = create_optimizer(
            init_lr=2e-5,
            num_warmup_steps=1_000,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01,
        )
        self.model.compile(optimizer=optimizer)

        # Train in mixed-precision float16
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # トレーニング前
        before_eval_loss = self.model.evaluate(tf_eval_dataset)

        # トレーニング開始
        self.model.fit(tf_train_dataset, validation_data=tf_eval_dataset)

        # トレーニング後
        after_eval_loss = self.model.evaluate(tf_eval_dataset)

        logger.info(f'Complete training! eval_loss: {before_eval_loss} -> {after_eval_loss}\n')
        return before_eval_loss > after_eval_loss

    def save_weights(self, path: str):
        self.model.save_weights(path)

    # 他のデコレーター
    def whole_word_masking_data_collator(self, features):
        wwm_probability = 0.2
        for feature in features:
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # ランダムに単語をマスクする
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = self.tokenizer.mask_token_id

        return default_data_collator(features, "tf")
