# ML Apprenticeship Take-Home Assignment

## Project Overview

This project implements a Sentence Transformer for generating sentence embeddings and extends it to a multi-task model capable of handling multiple NLP tasks. The primary objectives are to:

1. Implement a sentence transformer model to generate embeddings.
2. Expand it to a multi-task learning model for handling various NLP tasks.
3. Set layer-wise learning rates for training (Bonus Task).

----

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

----

Setup Instructions

Prerequisites

- Python 3.8 or above
- Virtualenv: Recommended for isolated environments
- Docker: Optional for containerization

1. Clone the Repository
   Clone the repository to your local machine:

git clone https://github.com/mouradImourad/ml_apprenticeship_take_home.git
cd ml_apprenticeship_take_home

2. Set Up a Virtual Environment

python3 -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

3. Install Dependencies

pip install -r requirements.txt

----

Usage Instructions

1. Training the Model
   To train the model, use the main.py script in the src/ directory. This will initialize the model and begin training:

python src/main.py --train

2. Running Inference
   After training, you can use the model to generate embeddings or handle other NLP tasks by running:

python src/main.py --inference --input "Your text here"

Example:

python src/main.py --inference --input "This is a sample sentence."

Expected output (example format):

{
"embedding": [0.1, 0.2, ...]
}

----

Testing
Unit tests are included in the tests/ directory. To run all tests, use:

pytest tests/

This validates that all components function as expected.

----

Docker Setup (Optional)

A Dockerfile is included for containerization. To build and run the Docker container:

1. Build the Docker Image:

docker build -t ml_apprenticeship_model .

2. Run the Docker Container:

docker run -it --rm ml_apprenticeship_model

This container includes all dependencies and environment configurations, allowing for easier deployment and reproducibility.

----

Advanced Training (Bonus Task)

For layer-wise learning rates, configure the learning rates in training_utils.py. Alternatively, you can pass a custom configuration file with layer-wise learning rate adjustments by specifying the path:

python src/main.py --train --config path/to/custom_config.json
