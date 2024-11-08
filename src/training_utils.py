import torch
from torch.optim import AdamW
from transformers import BertModel

def get_optimizer_with_layerwise_lr(model: BertModel, base_lr: float = 1e-5, layer_decay: float = 0.9):
    """
    Set up an optimizer with layer-wise learning rates for a BERT-based model.

    Args:
        model (BertModel): The model with a transformer backbone.
        base_lr (float): The base learning rate for the last layer.
        layer_decay (float): Multiplicative factor to decrease the learning rate for earlier layers.

    Returns:
        optimizer (torch.optim.Optimizer): Optimizer with specified learning rate for each layer.
    """
    # Store parameters and assign different learning rates per layer
    opt_parameters = []
    num_layers = len(model.encoder.layer)  # Number of transformer layers in the backbone
    
    # Decrease learning rate from last layer down to first layer
    for i in range(num_layers):
        layer = model.encoder.layer[num_layers - 1 - i]
        layer_lr = base_lr * (layer_decay ** i)  # Decrease learning rate for earlier layers
        
        opt_parameters += [
            {"params": layer.parameters(), "lr": layer_lr}
        ]
    
    # Set separate learning rates for the final classification and sentiment heads
    if hasattr(model, "classification_head"):
        opt_parameters += [{"params": model.classification_head.parameters(), "lr": base_lr}]
    
    if hasattr(model, "sentiment_head"):
        opt_parameters += [{"params": model.sentiment_head.parameters(), "lr": base_lr}]
    
    optimizer = AdamW(opt_parameters)
    return optimizer
