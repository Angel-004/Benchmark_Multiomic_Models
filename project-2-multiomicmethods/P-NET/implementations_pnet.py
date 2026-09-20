import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os  
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from palettable import wesanderson as wes
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.nn.functional as F

data_dir = "../data/TCGA-BRCA/"

# Add project path so Python can import modules
import sys

from MAIN.utils import *
from MAIN.train import *
#import MAIN.preprocess_functions

from MAIN.reactome import ReactomeNetwork
from MAIN.Pnet import MaskedLinear, PNET
from MAIN.utils import numpy_array_to_one_hot, get_gpu_memory
from MAIN.train import *
from MAIN.interpret import interpret, evaluate_interpret_save, visualize_importances

import pandas as pd
import gc
import pickle

def build_k_PNET_models(mRNA_exprs, DNAm_exprs, RPPA_exprs, df_all_meta, imputation_method,
                        batch_size = 256, lr_decay=0.2, num_epochs=200, learning_rate=0.1):
    """
    Build P-NET models across k folds using multi-omics data (mRNA, DNA methylation, RPPA).

    This function performs a k-fold training strategy, where each omics dataset is split into
    k folds. For each fold k:
        - The k-th fold is used as the test set.
        - The remaining k-1 folds are concatenated to form the training set.
        - mRNA and DNAm views undergo variance-based feature selection to keep the top
          2000 most variable features (RPPA is kept unchanged due to its smaller size).
        - The same selected features from the training set are enforced in the test set.
        - Multi-omic matrices (mRNA, DNAm, RPPA) are concatenated to build X_train and X_test.
        - Missing values are handled using one of three strategies:
              * 'drop'   : remove rows containing NaN values
              * 'median' : impute NaN using training-set medians for each feature
              * 'KNN'    : impute NaN using k-nearest neighbors fitted on the training set
        - Labels are extracted from the metadata table and mapped to integer categories.
        - A Reactome hierarchical network is constructed and used inside the PNET model.
        - The PNET model is trained on the training set, and evaluated on the test set.

    Parameters
    ----------
    mRNA_exprs : list of pandas.DataFrame
        List of length k containing mRNA expression matrices for each fold.
    DNAm_exprs : list of pandas.DataFrame
        DNA methylation matrices for each fold.
    RPPA_exprs : list of pandas.DataFrame
        RPPA protein expression matrices for each fold.
    df_all_meta : pandas.DataFrame
        Metadata table containing at least the PAM50 subtype labels.
    imputation_method : str
        Missing-value imputation strategy ('drop', 'median', or 'KNN').
    batch_size : int, optional
        Mini-batch size for training the PNET model.
    lr_decay : float, optional
        Learning-rate decay factor applied during training.
    num_epochs : int, optional
        Number of epochs to train each model.
    learning_rate : float, optional
        Initial learning rate.

    Returns
    -------
    models : list
        The list of k trained PNET models.
    train_loaders : list
        List of PyTorch dataloaders used during training, one per fold.
    test_loaders : list
        List of PyTorch dataloaders used during evaluation, one per fold.
    mapping : dict
        Mapping from integer class indices to PAM50 subtype names.
    model_features : list of Index
        List of feature sets selected for each fold (after variance selection).

    Notes
    -----
    - Feature selection is performed independently per fold using training data only.
    - Imputation is fitted exclusively on the training set to avoid data leakage.
    - Highly recommended to use a GPU or HPC environment because PNET training is
      computationally intensive.
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Lists that will collect models, dataloaders, and selected features across folds
    model_features = []
    models = []
    train_loaders = []
    test_loaders = []
    mapping = dict()

    # Load list of cancer-related genes used to construct the Reactome hierarchy
    # (from the PNET paper dataset)
    genes = pd.read_csv('../data/ext_data/cancer_genes.txt', header=0, delimiter='\t')

    # Select GPU if available (recommended), otherwise CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('mps')   # For Apple Silicon Macs, if desired

    # Iterate over k folds
    for k in range(0, len(mRNA_exprs)):

        # Extract the k-th fold as the test set for each omic
        mRNA_test, DNAm_test, RPPA_test = mRNA_exprs[k], DNAm_exprs[k], RPPA_exprs[k]

        # Concatenate all folds except k-th → training sets
        mRNA_train = pd.concat([mRNA_exprs[i] for i in range(len(mRNA_exprs)) if i != k])
        DNAm_train = pd.concat([DNAm_exprs[i] for i in range(len(DNAm_exprs)) if i != k])
        RPPA_train = pd.concat([RPPA_exprs[i] for i in range(len(RPPA_exprs)) if i != k])

        # Select top 2000 highest-variance features in mRNA and DNAm
        # (RPPA is not filtered because it has only 464 features)
        mRNA_train = mRNA_train[mRNA_train.var(axis=0).sort_values(ascending=False).head(2000).index]
        DNAm_train = DNAm_train[DNAm_train.var(axis=0).sort_values(ascending=False).head(2000).index]

        # Ensure test set has the same features selected from training
        mRNA_test = mRNA_test[mRNA_train.columns]
        DNAm_test = DNAm_test[DNAm_train.columns]

        # Concatenate omics views to form full multi-omics matrices
        X_train = pd.concat([mRNA_train, DNAm_train, RPPA_train], axis=1)
        X_test = pd.concat([mRNA_test, DNAm_test, RPPA_test], axis=1)

        # Store selected features for this fold
        model_features.append(X_train.columns)

        # Handle missing-value imputation according to user-selected strategy
        if (imputation_method == 'drop'):
            # Drop missing values.
            X_train = X_train.dropna()
            X_test = X_test.dropna()

        elif (imputation_method == 'median'):
            # Compute medians only from training set (no data leakage)
            medians = X_train.median()
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)

        elif (imputation_method == 'KNN'):
            # KNN imputer fitted only on the training set to avoid leakage
            imputer = KNNImputer(n_neighbors=6)
            X_ref = X_train.dropna()
            imputer.fit(X_ref)
            X_train = pd.DataFrame(
                imputer.transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            X_test = pd.DataFrame(
                imputer.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )

        # Extract PAM50 subtype labels
        y = df_all_meta['paper_BRCA_Subtype_PAM50'].astype('category')

        # Build mapping from integer code → subtype name
        mapping = dict(enumerate(y.cat.categories))
        y = y.cat.codes

        # Align labels to training/test samples
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]

        # Convert to numpy arrays for PyTorch compatibility
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        X_train = X_train.values
        X_test = X_test.values

        # Build Reactome hierarchical network (contains gene → pathway relationships)
        net = ReactomeNetwork(
            genes_of_interest=np.unique(list(genes['genes'].values)),
            n_levels=5
        )

        # Initialize PNET model using the constructed Reactome network
        model = PNET(
            reactome_network=net,
            input_dim=X_train.shape[1],
            output_dim=np.unique(y).shape[0],
            activation=nn.ReLU,
            dropout=0.1,
            filter_pathways=False,
            input_layer_mask=None
        )

        # Convert labels into one-hot format for training
        y_train = numpy_array_to_one_hot(y_train).astype(np.float32)
        y_test = numpy_array_to_one_hot(y_test).astype(np.float32)

        # Train the PNET model for this fold
        model, train_loader, test_loader = train(
            model, X_train, X_test, y_train, y_test, device,
            batch_size=batch_size, lr_decay=lr_decay,
            num_epochs=num_epochs, learning_rate=learning_rate,
            sparse=False
        )

        # Store fold outputs
        models.append(model)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return models, train_loaders, test_loaders, mapping, model_features

def commute_PNET_metrics(models,test_loaders):
    """
    Evaluates multiple trained models on their corresponding test DataLoaders.

    For each (model, test_loader) pair, the function:
      - runs evaluation and retrieves a metrics dictionary,
      - converts it into a structured pandas DataFrame,
      - extracts the global accuracy,
      - computes the mean F1-score over the four main classes (Basal, Her2, LumA, LumB).

    Returns:
      reports    : list of DataFrames containing classification metrics per model
      accuracies : list of global accuracies
      f1_scores  : list of mean F1-scores for the 4 primary classes
    """


    # Initialize lists to store evaluation outputs
    reports = []      # To store classification reports as DataFrames
    accuracies = []   # To store overall accuracy per model/fold
    f1_scores = []    # To store mean F1 score for the main classes

    # Loop over each trained model and its corresponding test DataLoader
    for model, test_load in zip(models, test_loaders):

        # Evaluate the model on the test set
        # 'evaluate' returns a dictionary with metrics for each class + overall metrics
        rep_dict = evaluate(model, test_load, device)  # ensure evaluation on CPU

        # Convert the evaluation dictionary to a pandas DataFrame
        # Transpose so that classes/metrics are rows/columns appropriately
        df = pd.DataFrame(rep_dict).transpose()
        df.index = ['Basal','Her2','LumA','LumB','accuracy','macro avg','weighted avg']
        # Store the classification report for this fold
        reports.append(df)

        # Compute mean F1 score over the main classes ('0', '1', '2', '3')
        f1_scores.append(np.mean(df.loc[['Basal','Her2','LumA','LumB'], ['f1-score']]))

        # Compute mean accuracy (overall) for this fold
        accuracies.append(np.mean(df.loc['accuracy']))
    return reports,accuracies,f1_scores