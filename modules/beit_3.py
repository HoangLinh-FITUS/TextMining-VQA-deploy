from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import os

def get_sentencepiece_model_for_beit3(sentencepiece_model):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(sentencepiece_model)
class Beit3Processing():
    def __init__(self, sentencepiece_model="beit3.spm", num_max_bpe_tokens=64, input_size=480):
        self.tokenizer = get_sentencepiece_model_for_beit3(sentencepiece_model)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    def _get_image(self, image):
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, image, question: str, data: dict):
        img = self._get_image(image)
        data["image"] = img
        tokens = self.tokenizer.tokenize(question)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        language_tokens, padding_mask, _ = self._get_text_segment(token_ids)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __call__(self, image, question: str):
        data = dict()
        self._get_image_text_example(image, question, data)
        return data