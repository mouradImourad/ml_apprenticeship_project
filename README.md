# ML Apprenticeship Take-Home Assignment

## Project Overview

This project implements a Sentence Transformer for generating sentence embeddings and extends it to a multi-task model capable of handling multiple NLP tasks. The primary objectives are to:

1. Implement a sentence transformer model to generate embeddings.
2. Expand it to a multi-task learning model for handling various NLP tasks.
3. Apply training considerations and transfer learning strategies.
4. Set layer-wise learning rates for training (Bonus Task).

---

## Project Structure

ml_apprenticeship_take_home/
├── src/ # Source code files
│ ├── transformer_model.py # Sentence Transformer implementation
│ ├── multi_task_model.py # Multi-Task Model implementation
│ ├── training_utils.py # Training utility functions
│ └── main.py # Script to run models
├── tests/ # Unit tests for each component
├── Dockerfile # Docker container setup
├── requirements.txt # Dependency list
└── README.md # Documentation

---

Setup Instructions

Prerequisites

- Python 3.8 or above
- Virtualenv: Recommended for isolated environments
- Docker: Optional for containerization

1. Clone the Repository
   Clone the repository to your local machine:

git clone https://github.com/mouradImourad/ml_apprenticeship_project.git
cd ml_apprenticeship_project

2. Set Up a Virtual Environment

python3 -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

3. Install Dependencies

pip install -r requirements.txt

---

Usage Instructions

Task 1: Sentence Transformer Model
The Sentence Transformer model encodes input sentences into fixed-length embeddings using a pre-trained BERT model.

To generate embeddings, use the following:

from src.transformer_model import SentenceTransformer

sentence_transformer = SentenceTransformer()
sentence = "I love coding."
embedding = sentence_transformer.encode(sentence)
print(f"Sentence Embedding (768-D vector): {embedding}")

Task 2: Multi-Task Model
The multi-task model builds on top of the sentence transformer by adding task-specific heads for classification tasks, such as:

Task A: Sentence Classification (e.g., classifying topics)
Task B: Sentiment Analysis
To process sentences for both tasks:

from src.multi_task_model import MultiTaskSentenceTransformer

multi_task_model = MultiTaskSentenceTransformer()
sentence = "This is a test sentence for multi-task learning."
output_a, output_b = multi_task_model(sentence)
print(f"Task A Output (Classification): {output_a}")
print(f"Task B Output (Sentiment): {output_b}")

Task 3: Training Considerations and Transfer Learning
The training utilities in training_utils.py allow for different training scenarios with the model:

Freezing the Entire Network:

This mode freezes the entire model to use it as a static feature extractor.
Useful for small datasets or limited computational resources.
Freezing Only the Transformer Backbone:

This keeps the core BERT model frozen while allowing task-specific heads to learn from the task data.
Suitable when task-specific adjustments are necessary without changing the language model.
Freezing Only One Task-Specific Head:

This allows only one task to update while the other remains fixed.
Useful when one task has less data or is more prone to overfitting.
Transfer Learning Setup:

Includes options for selective layer freezing and fine-tuning.
Example:
from src.training_utils import prepare_transfer_learning

model = MultiTaskSentenceTransformer()
prepare_transfer_learning(model, freeze_transformer=True, freeze_classification=False, freeze_sentiment=False)

---

Testing
Unit tests are included in the tests/ directory. To run all tests, use:

pytest tests/

This validates that all components function as expected.

---

Docker Setup (Optional)

A Dockerfile is included for containerization. To build and run the Docker container:

1. Build the Docker Image:

docker build -t ml_apprenticeship_model .

2. Run the Docker Container:

docker run -it --rm ml_apprenticeship_model

This container includes all dependencies and environment configurations, allowing for easier deployment and reproducibility.

---

Advanced Training (Bonus Task)

For layer-wise learning rates, configure the learning rates in training_utils.py. Alternatively, you can pass a custom configuration file with layer-wise learning rate adjustments by specifying the path:

python src/main.py --train --config path/to/custom_config.json
