# Output Layer Configurations for Binary Classification: Sin-gle-Neuron versus Dual-Neuron Approaches

This research project investigates the impact of using a single neuron versus two neurons in the output layer for binary classification tasks in neural networks.

## Research Question

In binary classification, the standard approach is to use a single neuron with sigmoid activation in the output layer. However, an alternative approach is to use two neurons with softmax activation, explicitly modeling both the positive and negative classes. This research aims to:

1. Experimentally compare the performance of both approaches
2. Identify advantages and disadvantages of each method
3. Determine if there are specific scenarios where one approach outperforms the other

## Methodology

1. Implement standard CNN architectures (ResNet) with both output configurations:
   - Single neuron with sigmoid activation
   - Two neurons with softmax activation
2. Train on standard benchmark datasets for binary classification
3. Compare performance metrics (accuracy, precision, recall, F1, ROC-AUC)
4. Analyze training dynamics, convergence speed, and generalization capability

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

See the scripts in the `src` directory for instructions on running experiments.

## Datasets

The experiments will use standard datasets widely accepted in the computer vision community, such as:
- Binary classification subsets of CIFAR-10

## Results and Findings

Experimental results will be documented in the `results` directory.

## License

This project is for research purposes only.
