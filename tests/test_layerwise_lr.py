import unittest
from src.training_utils import get_optimizer_with_layerwise_lr
from transformers import BertModel

class TestLayerwiseLearningRates(unittest.TestCase):
    def test_layerwise_lr(self):
        """Test that the optimizer is set up with decreasing learning rates per layer."""
        model = BertModel.from_pretrained("bert-base-uncased")
        optimizer = get_optimizer_with_layerwise_lr(model, base_lr=1e-5, layer_decay=0.9)
        
        # Extract learning rates for each layer
        lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        
        # Check if learning rates are decreasing
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i + 1], "Learning rates should decrease for earlier layers.")
            
if __name__ == "__main__":
    unittest.main()
