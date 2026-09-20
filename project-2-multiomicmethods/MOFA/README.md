# MOFA Notebook Environment Setup

To run the notebook **MOFA.ipynb**, follow these steps to set up the required environment.

## Prerequisites

- Python 3.8 or higher, here we used (3.11.5)
- pip (Python package installer)


## Environment Setup

1. Create a conda environment:
   ```bash
   conda create -n MOFA python=3.11.5 -y
   ```

2. Activate the environment:
   ```bash
   conda activate MOFA
   ```

3. Install the required packages:
   ```bash
   pip install -r MOFA/requirements_MOFA.txt
   ```

## Notes

- Ensure you have the necessary data files in the correct directory structure.
