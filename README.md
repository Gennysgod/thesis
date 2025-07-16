# Adaptive Financial Fraud Detection Experiment

## Project Structure
- `data_streams/`: Data generation modules
- `detectors/`: Concept drift detection algorithms
- `models/`: Neural network models for QuadCDD
- `evaluation/`: Metrics calculation and visualization
- `experiments/`: Experiment configuration and execution
- `results/`: Output results and figures

## Setup
```bash
pip install -r requirements.txt
python -m experiments.quadcdd_trainer --mode pretrain
python main.py
```

## Usage
1. Run pre-training for QuadCDD
2. Execute main experiment
3. Check results in `results/` directory