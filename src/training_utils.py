import torch
import torch.nn as nn

def freeze_entire_model(model):
    """Freeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = False


def freeze_transformer_backbone(model):
    """Freeze only the transformer (BERT) backbone, keep task heads trainable."""
    for param in model.model.parameters():  # Assuming 'model' is the BERT backbone
        param.requires_grad = False


def freeze_task_head(model, task='A'):
    """Freeze one task-specific head based on the task specified."""
    if task == 'A':
        for param in model.classification_head.parameters():
            param.requires_grad = False
    elif task == 'B':
        for param in model.sentiment_head.parameters():
            param.requires_grad = False


def prepare_transfer_learning(model, num_layers_to_freeze=6):
    """Freeze the first `num_layers_to_freeze` layers of the transformer backbone."""
    for i, layer in enumerate(model.model.encoder.layer):
        if i < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
