# P-NET Notebook Environment Setup

## Model Implementation

The P-NET model implemented in this project is based on the code from the GitHub repository [Barry8197/PNetTorch](https://github.com/Barry8197/PNetTorch).

Most of the code was copied here for convenience. The only modification we did was on the functions train() and evaluate() found in [MAIN/train.py](MAIN/train.py) for cross-validation.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Environment Setup

1. Create a conda environment:
   ```bash
   conda create -n pnet python=3.10
   ```

2. Activate the environment:
   ```bash
   conda activate pnet
   ```

3. Install the required packages:
   ```bash
   pip install -r P-NET/requirements_pnet.txt
   ```

## Notes

- Ensure you have the necessary data files in the correct directory structure.
