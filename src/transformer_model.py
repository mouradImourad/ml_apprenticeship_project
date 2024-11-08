import torch
from transformers import BertModel, BertTokenizer

class SentenceTransformer:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the Sentence Transformer with a pre-trained BERT model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode(self, sentence: str) -> torch.Tensor:
        """Encodes a sentence into a fixed-length embedding.

        Args:
            sentence (str): The sentence to encode.

        Returns:
            torch.Tensor: A tensor representing the sentence embedding.
        """
        
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128, padding=True)
        
    
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
        
        return cls_embedding.squeeze(0)  


if __name__ == "__main__":
    sentence_transformer = SentenceTransformer()
    sentence = "I love coding."
    embedding = sentence_transformer.encode(sentence)
    print(f"Sentence Embedding (768-D vector): {embedding}")
