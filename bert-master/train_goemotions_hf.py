"""GoEmotions training script using HuggingFace Transformers"""
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from torch.utils.data import Dataset

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

class GoEmotionsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    text = parts[0]
                    label_ids = parts[1].split(',')
                    label = int(label_ids[0]) if label_ids[0].isdigit() else 27
                    self.texts.append(text)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.texts)} examples from {data_path}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def main():
    data_dir = '/root/autodl-tmp/Campusmind/data/goemotions'
    output_dir = '/root/autodl-tmp/Campusmind/output/goemotions_hf'
    model_name = 'bert-base-cased'
    
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(EMOTIONS)
    )
    
    print("Loading datasets...")
    train_dataset = GoEmotionsDataset(os.path.join(data_dir, 'train.tsv'), tokenizer)
    eval_dataset = GoEmotionsDataset(os.path.join(data_dir, 'dev.tsv'), tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    results = trainer.evaluate()
    print(f"Results: {results}")
    
    trainer.save_model(os.path.join(output_dir, 'final_model'))
    print(f"Model saved to {output_dir}/final_model")

if __name__ == '__main__':
    main()
