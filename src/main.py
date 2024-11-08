from training_utils import (
    freeze_entire_model,
    freeze_transformer_backbone,
    freeze_task_head,
    prepare_transfer_learning
)
from src.multi_task_model import MultiTaskSentenceTransformer

# Initialize model
model = MultiTaskSentenceTransformer()

# Apply any freezing scenario you need
freeze_entire_model(model)               # Scenario 1
freeze_transformer_backbone(model)       # Scenario 2
freeze_task_head(model, task='A')        # Scenario 3
prepare_transfer_learning(model, num_layers_to_freeze=6)  # Transfer Learning scenario
