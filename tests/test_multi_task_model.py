import unittest
import torch
from src.multi_task_model import MultiTaskSentenceTransformer

class TestMultiTaskSentenceTransformer(unittest.TestCase):
    def setUp(self):
        """Set up the MultiTaskSentenceTransformer instance for tests."""
        self.model = MultiTaskSentenceTransformer()
        
    def test_output_shapes(self):
        """Test that the model produces outputs with the correct shapes for each task."""
        sentence = "This is a test sentence."
        output_a, output_b = self.model(sentence)
        
        self.assertEqual(output_a.shape, (1, 3))  
        
        self.assertEqual(output_b.shape, (1, 2))  

    def test_output_values(self):
        """Test that the model outputs reasonable values for different sentences."""
        sentence_positive = "I love machine learning!"
        sentence_negative = "I hate bugs in my code."
        
        output_a_pos, output_b_pos = self.model(sentence_positive)
        output_a_neg, output_b_neg = self.model(sentence_negative)
        
        self.assertFalse(torch.isnan(output_a_pos).any())
        self.assertFalse(torch.isnan(output_b_pos).any())
        self.assertFalse(torch.isnan(output_a_neg).any())
        self.assertFalse(torch.isnan(output_b_neg).any())

if __name__ == "__main__":
    unittest.main()
