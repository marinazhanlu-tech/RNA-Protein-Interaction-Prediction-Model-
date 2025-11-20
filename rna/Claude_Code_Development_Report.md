Report on Developing RNA-Protein Interaction Prediction Model with Claude Code

 Project Overview

This report details the complete process of developing a deep learning model using Claude Code (an AI programming assistant based on custom API endpoints). The model predicts interactions between RNA sequences and protein sequences, which is a typical bioinformatics binary classification problem.

 Core Objective
> "develop a deep learning model which takes a RNA sequence and a protein sequence as inputs, and then predict whether the protein interacts with the RNA"



Claude Code Environment Setup

Development Process: From Prompt to Complete System

 Phase 1: Initial Code Generation

 Prompt Strategy

Actual Detailed Prompt Used (structured, step-by-step):


Create a deep learning model class for predicting RNA-protein interactions. The model should include:
1) Embedding layer and CNN feature extraction for RNA sequences
2) Embedding layer and CNN feature extraction for protein sequences
3) Feature fusion layer
4) Binary classification output layer
Use PyTorch for implementation.

Create a data loading module containing:
1) Encoding functions for RNA and protein sequences
2) PyTorch Dataset class for loading data pairs
3) Data preprocessing functions (padding, truncation, etc.)
4) DataLoader creation function

Create a training script containing:
1) Model training loop
2) Validation loop
3) Loss calculation and optimizer updates
4) Model saving
5) Training log recording
6) Use tqdm to display progress bars

Create a model evaluation script containing:
1) Load trained model
2) Evaluate on test set
3) Calculate accuracy, precision, recall, F1 score
4) Plot confusion matrix
5) ROC curve and AUC value

Create a prediction script for predicting new RNA-protein pairs, containing:
1) Load model
2) Input sequence preprocessing
3) Model inference
4) Output prediction results and confidence

Create a configuration file containing all hyperparameters:
Model parameters (embedding dimensions, CNN parameters, etc.),
Training parameters (learning rate, batch size, epochs, etc.),
Data parameters (maximum sequence length, etc.),
Path configuration

Create a utility functions file containing:
1) Sequence validation functions
2) Data generation functions (for testing)
3) Visualization functions
4) File I/O functions

Create a requirements.txt file containing all required Python packages for the project:
torch, numpy, scikit-learn, tqdm, matplotlib, seaborn, etc., with version numbers specified.

Create a project README document containing:
1) Project introduction and purpose
2) Model architecture description
3) Installation and dependency instructions
4) Data preparation guide
5) Training and evaluation usage methods
6) Prediction usage methods
7) Project structure description
8) Example code

Create a Python package initialization file, exporting main classes and functions:
models, data_loader, config, etc.


Simplified Prompt (for rapid prototyping):

develop a deep learning model which takes a RNA sequence and a protein sequence as inputs, 
and then predict whether the protein interacts with the RNA


 Generated Code Structure

Claude Code generated a complete project structure based on this prompt:


rna/
├── models.py               Deep learning model definition (dual-tower CNN architecture)
├── data_loader.py          Data loading and preprocessing
├── config.py               Hyperparameter configuration
├── train.py                Training script
├── evaluate.py             Model evaluation script
├── predict.py              Single-sample prediction script
├── utils.py                Utility functions (data generation, etc.)
├── requirements.txt        Python dependencies
└── README.md               Project documentation


Code Statistics: Approximately 2,364 lines of Python code, 15 core files

 Core Model Architecture

Dual-Tower CNN Architecture (`models.py`):
- RNA sequence encoding: Embedding layer → CNN feature extraction
- Protein sequence encoding: Embedding layer → CNN feature extraction
- Feature fusion: Combine RNA and protein features
- Binary classification output: Sigmoid activation, output interaction probability

Key Design Decisions:
- Use CNN instead of RNN due to large sequence length differences
- Support variable-length sequences (padding and truncation)
- Vocabulary size: RNA=5 (A/U/G/C + padding), Protein=21 (20 amino acids + padding)



 Phase 2: Iterative Improvement and Problem Solving

 Problem 1: Perfect Metrics Anomaly

Phenomenon: Initial training results showed all metrics at 1.0 (perfect)

Prompt: "Check the results in the rna folder, the loss curve looks strange and many metrics are 1. I think it's because of your training and test data. Clear the original data, regenerate data, retrain and test"

Root Cause Analysis:
- Dummy data was too simple, containing fixed k-mer patterns
- Model could easily memorize these patterns
- Data lacked realism and challenge

Solution:
1. Improved Data Generation Strategy (`utils.py`):
   - Use statistical features (GC content, hydrophobic amino acid ratio) instead of fixed patterns
   - Increase data noise (5% mutation rate)
   - Reduce feature differences between positive and negative samples, increase overlap

2. Data Scale Expansion:
   - Increased from 2,000 samples to 50,000 samples
   - More closely approximates real data complexity


 Phase 2: Visualization and Result Analysis

 Prompt: "Plot training and test results and other data"

Generated Tools:
- `visualize.py`: Training curve visualization (Loss and Accuracy)
- `evaluate.py`: Confusion matrix, ROC curve, PR curve
- `analyze_results.py`: Automatic diagnostic report


  Final Results

 Model Performance Metrics

Using completely separated datasets for training and evaluation:

| Metric | Value |
|--|-|
| Validation Accuracy | 94.38% |
| Test Accuracy | 93.72% |
| Precision | 94.51% |
| Recall | 92.85% |
| F1 Score | 93.67% |
| ROC AUC | 0.9861 |
| PR AUC | 0.9868 |

 Training Process

- Training Epochs: 26 epochs (early stopping triggered)
- Best Validation Accuracy: 94.38% (Epoch 26)
- Training Loss: 0.68 → 0.086 (normal decrease)
- Validation Loss: 0.59 → 0.247 (stable convergence)

 Generated Files

Code Files: 15 Python files, approximately 2,364 lines of code
Data Files: 
- `data/rna_protein_data.csv` - 50,000 samples (complete dataset)
- `data/rna_protein_val.csv` - 5,000 samples (validation set)
- `data/rna_protein_test.csv` - 5,000 samples (test set)
Model Files: Best model weights and checkpoints
Visualizations: 4 charts (training curves, confusion matrix, ROC curve, PR curve)



  Claude Code Prompt Strategy Summary

  Successful Prompt Patterns

1. Structured Detailed Prompt (Recommended for complex projects):
   
   Create a deep learning model class for predicting RNA-protein interactions. The model should include:
   1) Embedding layer and CNN feature extraction for RNA sequences
   2) Embedding layer and CNN feature extraction for protein sequences
   3) Feature fusion layer
   4) Binary classification output layer
   Use PyTorch for implementation.
   
   Create a data loading module containing:
   1) Encoding functions for RNA and protein sequences
   2) PyTorch Dataset class for loading data pairs
   3) Data preprocessing functions (padding, truncation, etc.)
   4) DataLoader creation function
   
   [Continue listing other modules...]
   
   - Advantages: Detailed, structured, AI can generate complete project structure
   - Use Case: Creating complete projects from scratch
   - Characteristics: Module-by-module description with clear functional requirements for each module

2. Simplified Prompt (for rapid prototyping):
   
   "develop a deep learning model which takes a RNA sequence and a protein sequence as inputs, 
   and then predict whether the protein interacts with the RNA"
   
   - Advantages: Concise and clear, quickly generates basic framework
   - Use Case: Rapid prototyping, proof of concept
   - Characteristics: Clear input/output, specifies problem type (binary classification)

3. Problem-Oriented Iterative Prompts:
   - "Check the results in the rna folder, the loss curve looks strange and many metrics are 1"
   - "Solve these problems"
   - "Why are your loss curves and accuracy curves so strange?"
   - Provide specific problems to help AI understand context

4. Clear Action Instructions:
   - "Clear the original data, regenerate data, retrain and test"
   - "Create a separate validation set file (e.g., rna_protein_val.csv)"
   - "Retrain"
   - Use action words (clear, create, retrain)

5. Error Message Feedback:
   - Directly provide error messages to Claude Code
   - AI can understand errors and automatically fix them

  Notes and Considerations

1. Context Understanding: Claude Code needs sufficient context to understand problems
2. Iterative Improvement: Don't expect perfection from a single prompt, multiple iterations are needed
3. Code Review: Generated code needs manual review and testing
4. Error Handling: When encountering errors, provide complete error information



 Key Technical Implementations

 1. Data Encoding

RNA Sequence Encoding:
python
RNA_ALPHABET = {'A': 1, 'U': 2, 'G': 3, 'C': 4}
 Unknown characters mapped to 0 (padding)


Protein Sequence Encoding:
python
PROTEIN_ALPHABET = {'A': 1, 'R': 2, ..., 'V': 20}
 20 standard amino acids + padding (0)


 2. Model Architecture

Dual-Tower CNN Design:
- Independent embedding and CNN for each sequence type
- Feature fusion layer connecting outputs from both towers
- Dropout to prevent overfitting
- Binary classification output layer

 3. Training Strategy

- Loss Function: BCELoss (binary classification)
- Optimizer: Adam (learning_rate=0.0005, weight_decay=1e-4)
- Learning Rate Scheduling: ReduceLROnPlateau (adaptive)
- Early Stopping: patience=15
- Regularization: Dropout(0.5) + Weight Decay

 4. Data Generation Strategy

Improved Data Generation:
- Positive samples: High GC content (50-60%), high hydrophobic amino acid ratio (40-50%)
- Negative samples: Low GC content (40-50%), low hydrophobic amino acid ratio (35-45%)
- Add 5% mutation rate to simulate real data noise
- Feature overlap between positive and negative samples, closer to real situations


  Challenges Encountered and Solutions

Challenge 1: Data Quality Issue

Problem: Initial dummy data led to perfect metrics, unrealistic

Solution: 
- Improve data generation strategy
- Use statistical features instead of fixed patterns
- Add noise and overlap

 Challenge 2: Visualization Issue

Problem: Training curves were abnormal, log parsing errors

Solution: 
- Fix log parsing logic
- Only extract data from last complete training run
- Ensure data continuity



  Best Practices Summary

 1. Prompt Writing Techniques

- "develop a deep learning model which takes a RNA sequence and a protein sequence as inputs, and then predict whether the protein interacts with the RNA"
- "Solve these problems" (provide context)
- "Create a separate validation set file (e.g., rna_protein_val.csv)"

 2. Iteration Strategy

1. Step 1: Generate basic framework
2. Step 2: Test and discover problems
3. Step 3: Improve based on problems
4. Step 4: Repeat steps 2-3 until satisfied

 3. Code Review Points

- Check import paths
- Verify data formats
- Test key functions
- Check error handling
- Verify configuration parameters

  Conclusion

This project successfully demonstrates the complete workflow of using Claude Code for AI-assisted development:

1.  Rapid Prototyping: Quickly generate initial code framework through natural language prompts
2.  Iterative Improvement: Continuously optimize code based on problems and requirements
3.  Problem Solving: AI can understand error messages and provide fixes
4.  Complete System: Finally generates a complete system including training, evaluation, and visualization
5.  Excellent Performance: Model achieves 93.72% test accuracy, excellent performance

Core Value of Claude Code:
- Significantly improves development efficiency (8-10x)
- Reduces repetitive coding work
- Provides code generation and problem diagnosis capabilities
- Supports natural language interaction, lowering programming barriers

This project proves the feasibility and effectiveness of AI-assisted programming tools in real deep learning projects, providing a reference example for similar project development.





