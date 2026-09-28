import torch
import torch.nn as nn
from transformers import AutoModel

class CodeQualityClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)