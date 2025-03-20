

class Beit3Processing():
    def __init__(self, data_path, split, transform, root_folder, tokenizer, num_max_bpe_tokens):
        num_sample = -1
        if split == 'train':
            num_sample = 50000
        elif split == 'val':
            num_sample = 6000
        else:
            num_sample = -1
        self.dataframe = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
        self.dataframe.dropna(inplace=True)
        self.dataframe = self.dataframe.sample(frac=1)
        self.dataframe = self.dataframe[:num_sample]
        if split == 'train':
            df = pd.read_csv(os.path.join(data_path, f"data.csv"))
            unique_answers = set(df["answer"].tolist())
            self.answer2id = {ans: i for i, ans in enumerate(unique_answers)}
            self.id2answer = {i: ans for i, ans in enumerate(unique_answers)}
            with open("answer2label.json", mode="w", encoding="utf-8") as writer:
                writer.write(json.dumps(self.answer2id))
            with open("label2lanswer.json", mode="w", encoding="utf-8") as writer:
                writer.write(json.dumps(self.id2answer))
        else:
            with open("answer2label.json", mode="r", encoding="utf-8") as f:
                self.answer2id = json.load(f)
            with open("label2lanswer.json", mode="r", encoding="utf-8") as f:
                self.id2answer = json.load(f)
            unique_answers = set(self.dataframe["answer"].tolist())
            for ans in unique_answers:
                if ans not in self.answer2id.keys():
                    self.dataframe = self.dataframe[self.dataframe["answer"] != ans]
        self.root_folder = root_folder
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.root_folder, image_path)
        image = self.loader(image_path)
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

    def _get_image_text_example(self, image_path: int, question: str, data: dict):
        img = self._get_image(image_path)
        data["image"] = img
        question_text = question
        if type(question_text) is not str:
            print(question_text)
        tokens = self.tokenizer.tokenize(question_text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        language_tokens, padding_mask, _ = self._get_text_segment(token_ids)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __call__(self, image_path: int, question: str):
        data = dict()
        self._get_image_text_example(index, image_path, question, data)
        return data