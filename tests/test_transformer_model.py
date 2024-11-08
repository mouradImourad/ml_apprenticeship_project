import unittest
import torch
from src.transformer_model import SentenceTransformer
import torch.nn.functional as F

class TestSentenceTransformer(unittest.TestCase):
    def setUp(self):
        """Set up the SentenceTransformer instance for tests."""
        self.model = SentenceTransformer()

    def test_embedding_shape(self):
        """Test that the output embedding has the correct shape."""
        sentence = "This is a test sentence."
        embedding = self.model.encode(sentence)
        
        self.assertEqual(embedding.shape, (768,))

    def test_similarity_between_similar_sentences(self):
        """Test that similar sentences produce similar embeddings."""
        sentence1 = "I love machine learning."
        sentence2 = "I enjoy machine learning."
        
        embedding1 = self.model.encode(sentence1)
        embedding2 = self.model.encode(sentence2)
        
        cosine_similarity = F.cosine_similarity(embedding1, embedding2, dim=0)
        
        self.assertGreater(cosine_similarity, 0.8)

    def test_relative_dissimilarity(self):
        """Check that unrelated sentences have lower similarity compared to similar sentences."""
        sentence1 = "I love machine learning."
        sentence2 = "Quantum physics is challenging."
        
        similar_sentence = "I enjoy machine learning."
        
        embedding1 = self.model.encode(sentence1)
        embedding2 = self.model.encode(sentence2)
        embedding_similar = self.model.encode(similar_sentence)
        
        similarity_to_similar = F.cosine_similarity(embedding1, embedding_similar, dim=0)
        similarity_to_dissimilar = F.cosine_similarity(embedding1, embedding2, dim=0)
        
        self.assertGreater(similarity_to_similar, similarity_to_dissimilar)

if __name__ == "__main__":
    unittest.main()
