from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch


class YahooAnswersDataset(Dataset):
    def __init__(self, split, tokenizer_name='bert-base-uncased', max_length=512):
        # Load the dataset from Hugging Face datasets
        self.dataset = load_dataset("yahoo_answers_topics", split=split)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Maximum length for padding/truncation
        self.max_length = max_length

        # Prepare classes and targets
        self.classes = self.dataset.features['topic'].names
        self.targets = [example['topic'] for example in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract the item
        item = self.dataset[idx]

        # Tokenize the text
        inputs = self.tokenizer(
            item['question_title'] + ' ' + item['question_content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Convert input_ids and attention_mask to tensors
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # Get the label
        label = torch.tensor(item['topic'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }