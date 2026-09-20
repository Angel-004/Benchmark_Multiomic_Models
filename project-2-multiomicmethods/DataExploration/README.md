# Baseline Notebook Environment Setup

To run the notebook **DataExploration.ipynb**, follow these steps to set up the required environment.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Environment Setup

1. Create a conda environment:
   ```bash
   conda create -n DataExploration python=3.10 -y
   ```

2. Activate the environment:
   ```bash
   conda activate DataExploration
   ```

3. Install the required packages:
  ```bash
   pip install -r DataExploration/requirements_data_exploration.txt
   ```

## Notes

- Ensure that you have placed the omics data files (`mRNA.pkl`, `DNAm.pkl`, `RPPA.pkl`) in the `../data/TCGA-BRCA/` directory relative to the notebook.

- Run the notebook cells in order.
 
No additional configuration is needed.
