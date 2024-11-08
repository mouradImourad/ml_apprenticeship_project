from src.training_utils import get_optimizer_with_layerwise_lr
from src.multi_task_model import MultiTaskSentenceTransformer

# Initialize model and optimizer
model = MultiTaskSentenceTransformer()
optimizer = get_optimizer_with_layerwise_lr(model.model, base_lr=1e-5, layer_decay=0.9)

# Example training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        # Forward pass
        output_a, output_b = model(batch["input_text"])
        # Compute loss and backpropagate
        loss = compute_loss(output_a, output_b, batch)
        loss.backward()
        optimizer.step()
