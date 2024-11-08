import unittest
import torch
from src.multi_task_model import MultiTaskSentenceTransformer
from src.training_utils import (
    freeze_entire_model,
    freeze_transformer_backbone,
    freeze_task_head,
    prepare_transfer_learning
)

class TestTrainingUtils(unittest.TestCase):
    def setUp(self):
        """Set up the model instance for testing."""
        self.model = MultiTaskSentenceTransformer()

    def test_freeze_entire_model(self):
        """Test that all parameters are frozen when entire model is frozen."""
        freeze_entire_model(self.model)
        all_frozen = all(param.requires_grad == False for param in self.model.parameters())
        self.assertTrue(all_frozen, "Not all parameters were frozen in the model.")

    def test_freeze_transformer_backbone(self):
        """Test that only transformer backbone is frozen."""
        freeze_transformer_backbone(self.model)
        backbone_frozen = all(param.requires_grad == False for param in self.model.model.parameters())
        head_unfrozen = any(param.requires_grad == True for param in self.model.classification_head.parameters()) and \
                        any(param.requires_grad == True for param in self.model.sentiment_head.parameters())
        self.assertTrue(backbone_frozen, "Not all transformer backbone parameters were frozen.")
        self.assertTrue(head_unfrozen, "Task heads should remain trainable.")

    def test_freeze_task_head_classification(self):
        """Test that only the classification head is frozen."""
        freeze_task_head(self.model, task='A')
        classification_head_frozen = all(param.requires_grad == False for param in self.model.classification_head.parameters())
        sentiment_head_unfrozen = any(param.requires_grad == True for param in self.model.sentiment_head.parameters())
        self.assertTrue(classification_head_frozen, "Classification head was not frozen.")
        self.assertTrue(sentiment_head_unfrozen, "Sentiment head should remain trainable.")

    def test_freeze_task_head_sentiment(self):
        """Test that only the sentiment head is frozen."""
        freeze_task_head(self.model, task='B')
        sentiment_head_frozen = all(param.requires_grad == False for param in self.model.sentiment_head.parameters())
        classification_head_unfrozen = any(param.requires_grad == True for param in self.model.classification_head.parameters())
        self.assertTrue(sentiment_head_frozen, "Sentiment head was not frozen.")
        self.assertTrue(classification_head_unfrozen, "Classification head should remain trainable.")

    def test_prepare_transfer_learning(self):
        """Test that the first few layers of the transformer backbone are frozen."""
        num_layers_to_freeze = 6
        prepare_transfer_learning(self.model, num_layers_to_freeze=num_layers_to_freeze)
        frozen_layers = sum(1 for i, layer in enumerate(self.model.model.encoder.layer[:num_layers_to_freeze])
                            if all(param.requires_grad == False for param in layer.parameters()))
        unfrozen_layers = sum(1 for i, layer in enumerate(self.model.model.encoder.layer[num_layers_to_freeze:])
                              if any(param.requires_grad == True for param in layer.parameters()))
        self.assertEqual(frozen_layers, num_layers_to_freeze, "Not all intended layers were frozen.")
        self.assertTrue(unfrozen_layers > 0, "Layers after frozen layers should remain trainable.")

if __name__ == "__main__":
    unittest.main()
