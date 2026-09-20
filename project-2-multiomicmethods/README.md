# Benchmarking different multi-omic models for breast cancer subtypes classification


## Overview
  This project seeks to study novel multi-omic methods, whose main features include the ability to process data from different biomolecular layers (e.g. mRNA, DNAm, RPPA) simultaneously. In particular, we implement and compare three of them: 
  - P-NET
  - IntegrAO
  - MOFA

while setting a classical logistic regression pipeline as a baseline. We consider the context of breast cancer, with data taken from the TCGA-BRCA dataset, and evaluate these models in a tumour subtype classification problem. The metrics we use for evaluation include performance (in terms of accuracy and F1-score) and interpretability of the model.


## Structure
The repository is structured as follows:
```
├── data
│   ├── cancer_genes.txt
│   ├── DatasetLink.md                       <- Contains the link to download the dataset. You have the link below as well
│   └── TCGA-BRCA                            <- Please download the dataset in this folder 
│ 
├── DataExploration 
│   ├── DataExploration.ipynb 
│   ├── requirements_data_exploration.txt
│   └── README.md
├── Baseline 
│   ├── Baseline.ipynb
│   ├── requirements_baseline.txt
│   ├── implementations_Baseline.py 
│   └── README.md                    
├── MOFA
│   ├── MOFA.ipynb
│   ├── requirements_baseline.txt
│   ├── implementations_MOFA.py 
│   └── README.md                          
├── IntegrAO
│   ├── IntegrAO.ipynb
│   ├── requirements_integrao.txt
│   ├── implementations_IntegrAO.py 
│   └── README.md                          
├── P-NET
│   ├── P-NET.ipynb
│   ├── reactome                           <- P-NET implementation code that we used for this project
│   ├── requirements_pnet.txt
│   ├── implementations_pnet.py 
│   └── README.md                       
│
├── utility.py                             <- General utility helper functions
│
├── Project_Report.pdf                     <- Copy of the report for the project
│
└── README.md                              <- Here we are ! 
```
In particular, for each model we have a corresponding folder with this structure:
```
├── Model 
│   ├── Model.ipynb                        <- Notebook presenting the results
│   ├── implementations_Model.py           <- python file with helper functions specific to the model
│   ├── requirements_Model.txt             <- .txt file with the necessary libraries to run the code
│   └── README.md                          <- .md file explaining the specificities of the model and how to run the notebook.
```


## Data
For all of our results we need the following dataset, obtained from the TCGA-BRCA database: 

https://drive.google.com/file/d/17aOoCjF9AFo7Y_UjYlV4wNWx-SDHac_o/view?usp=sharing

It needs to be extracted in [data/TCGA-BRCA](data/TCGA-BRCA). The folder should contains 3 pickle files. More precisely, it should look like

```
├── data
│   ├── cancer_genes.txt 
│   └── TCGA-BRCA
│       ├── DNAm.pkl
│       ├── mRNA.pkl
│       └── RPPA.pkl
```

## Libraries
The following Python libraries, available on the Python Package Index, are used across the project:

- captum
- integrao
- matplotlib
- matplotlib-venn
- networkx
- numpy
- palettable
- pandas
- plotly
- scikit-learn
- scipy
- seaborn
- skunk
- snfpy
- torch
- typing_extensions
- umap-learn

You can install them using ```pip install -r requirements_*.txt``` for each module's requirements file.

## Team members

  -  Igor Jomaron (igor.dejomaron2000@gmail.com)
  -  Fabien Donnet-Monay (fabien.donnet-monay@epfl.ch)
  -  Ángel Zhang (angel.zhanghuang@epfl.ch)

Please feel free to contact us should you have any inquiry regarding the project








