import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes_a=3, num_classes_b=2):
        """Initialize the Multi-Task Sentence Transformer model."""
        super(MultiTaskSentenceTransformer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
        self.classification_head = nn.Linear(self.model.config.hidden_size, num_classes_a)
        
        self.sentiment_head = nn.Linear(self.model.config.hidden_size, num_classes_b)

    def forward(self, sentence: str):
        """Process a sentence and produce outputs for multiple tasks."""
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():  
            outputs = self.model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
        
        output_a = self.classification_head(cls_embedding)
        
        output_b = self.sentiment_head(cls_embedding)
        
        return output_a, output_b


if __name__ == "__main__":
    multi_task_model = MultiTaskSentenceTransformer()
    sentence = "This is a test sentence for multi-task learning."
    output_a, output_b = multi_task_model(sentence)
    
    print(f"Task A Output (Classification): {output_a}")
    print(f"Task B Output (Sentiment): {output_b}")
