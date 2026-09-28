import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from model import CodeQualityClassifier

class CodeDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["code"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

def train():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = CodeDataset("data/code_quality_dataset.jsonl", tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = CodeQualityClassifier(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    print("Starting 1 epoch of training...")
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(outputs, batch["label"])
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), "model.pt")
    print("Training complete! Model weights saved as model.pt")

if __name__ == "__main__":
    train()