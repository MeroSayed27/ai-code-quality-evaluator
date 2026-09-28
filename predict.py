import torch
from transformers import AutoTokenizer
from model import CodeQualityClassifier

def evaluate_code(code_snippet):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = CodeQualityClassifier(model_name)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(outputs, dim=1).item()

    return "Good Code" if prediction == 1 else "Hallucinated / Bad Code"

if __name__ == "__main__":
    sample_code = "def add(a, b):\n    return a + b"
    result = evaluate_code(sample_code)
    print(f"Evaluation Result: {result}")