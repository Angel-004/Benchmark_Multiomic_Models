#import library
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from palettable import wesanderson as wes
import mofapy2
from mofapy2.run.entry_point import entry_point
import mofax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def MOFA_format(expr, view_name):
    """
    Convert an omics expression matrix into a long-format DataFrame compatible with MOFA.

    Parameters
    ----------
    expr : pandas.DataFrame
        Omics expression matrix where rows represent samples and columns represent features.
    view_name : str
        Name of the view (e.g., the type of omics data) to be added as a new column.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns:
            - 'sample': sample identifier (from the original row index)
            - 'view': name of the view provided
            - 'feature': original column name (feature)
            - 'value': expression value for that feature in that sample
    """
    df_long = (
        expr
        .assign(
            sample=expr.index, 
            view=view_name  
        )
        .melt(
            id_vars=['sample', 'view'], 
            var_name='feature',  
            value_name='value'  
        )
    )
    return df_long




def build_k_MOFA_models(mRNA_exprs, DNAm_exprs, RPPA_exprs):
    
    """
    Train K MOFA models using K-fold cross-validation for multiple omics datasets.

    Parameters
    ----------
    mRNA_exprs : list of pandas.DataFrame
        List of mRNA expression matrices, one per fold.
    DNAm_exprs : list of pandas.DataFrame
        List of DNA methylation expression matrices, one per fold.
    RPPA_exprs : list of pandas.DataFrame
        List of RPPA protein expression matrices, one per fold.

    Notes
    -----
    - Assumes each element in the lists corresponds to one fold of the dataset.
    - For each fold k, that fold is left out as test data and the remaining folds are used for training.
    - mRNA and DNAm matrices are filtered to the top 2000 most variable features.
    - RPPA matrices are used as-is (fewer than 2000 features).
    - Each omics matrix is converted to MOFA-compatible long format using `MOFA_format`.
    - All views are combined into a single DataFrame for MOFA input.
    - Each model is initialized, trained, and saved separately for each fold.
    """
    # Iterate over all folds (K-fold cross-validation)
   
    for k in range(0,len(mRNA_exprs)):
        
        # -----------------------------
        # Prepare training sets: all folds except the current fold k
        # -----------------------------
        mRNA_train = pd.concat([mRNA_exprs[i] for i in range(len(mRNA_exprs)) if i != k])
        DNAm_train = pd.concat([DNAm_exprs[i] for i in range(len(DNAm_exprs)) if i != k])
        RPPA_train = pd.concat([RPPA_exprs[i] for i in range(len(RPPA_exprs)) if i != k])
        
        # -----------------------------
        # Feature selection: select top 2000 most variable features for mRNA and DNAm
        # RPPA is kept as-is
        # -----------------------------
        mRNA_train = mRNA_train[mRNA_train.var(axis=0).sort_values(ascending=False).head(2000).index]
        DNAm_train = DNAm_train[DNAm_train.var(axis=0).sort_values(ascending=False).head(2000).index]
   
        
        # -----------------------------
        # Convert omics matrices to MOFA-compatible long format
        # Each row represents a single measurement (cell/view)
        # -----------------------------
        M_mRNA_train = MOFA_format(mRNA_train,'mRNA')
        M_DNAm_train = MOFA_format(DNAm_train,'DNAm')
        M_RPPA_train = MOFA_format(RPPA_train,'RPPA')

        # Combine all views into a single long-format DataFrame
        df_all_train = pd.concat([M_mRNA_train,M_DNAm_train,M_RPPA_train],axis=0, ignore_index=True)

        
        # -----------------------------
        # Initialize MOFA model
        # -----------------------------
        ent = entry_point()
        
       # Set data options
        ent.set_data_options(
            scale_groups=False,  # do not scale samples/groups
            scale_views=False,   # do not scale views
        )
        # Provide long-format DataFrame to MOFA
        ent.set_data_df(df_all_train,likelihoods = ["gaussian","gaussian","gaussian"])
        
        # Set model options
  
        ent.set_model_options(
            factors=15,             # number of latent factors
            spikeslab_weights=True, # spike-and-slab prior for weights
            spikeslab_factors=False,
            ard_factors=True,       # automatic relevance determination for factors
            ard_weights=True        # automatic relevance determination for weights
        )
    
    
        # Set training options
        ent.set_train_options(
            convergence_mode="slow",
            dropR2=0.001,
            gpu_mode=False,
            seed=42
        )
        
        # Build and train the MOFA model
        ent.build()
        ent.run()
        
        # Save the trained model for this fold
        filename = f"mofa_model_balanced_2000_fold{k}.hdf5"
        ent.save(filename)

        
        
        
def train_test_log_reg_on_MOFA(mRNA_exprs, DNAm_exprs, RPPA_exprs, model_list, df_all_meta):
    """
    Train and evaluate logistic regression models using factors computed from MOFA models.

    Description:
    For each MOFA model in the list, this function:
    1. Constructs training and test sets from the corresponding omics data.
    2. Computes factors for train and test sets by projecting data onto the MOFA weight matrix.
       - Handles NaN values by solving least squares for each sample independently.
    3. Selects only factors significantly correlated with tumor subtypes.
    4. Trains a logistic regression model on the selected factors to predict tumor subtypes.
    5. Computes and stores performance metrics (accuracy, F1 score, classification report) for 
       both training and test sets.

    Inputs:
    - mRNA_exprs : list of pandas DataFrames
        Expression matrices for mRNA data split into folds (rows = samples, columns = features).
    - DNAm_exprs : list of pandas DataFrames
        Expression matrices for DNA methylation data split into folds.
    - RPPA_exprs : list of pandas DataFrames
        Expression matrices for RPPA (protein) data split into folds.
    - model_list : list of trained MOFA models
        MOFA models used to compute latent factors.
    - df_all_meta : pandas DataFrame
        Metadata for all samples, including the 'paper_BRCA_Subtype_PAM50' column used as the target.

    Outputs:
    - train_accuracies : list of floats
        Accuracy scores on the training sets for each MOFA model.
    - train_f1_scores : list of floats
        Macro F1 scores on the training sets.
    - train_reports : list of pandas DataFrames
        Classification reports (precision, recall, F1) on the training sets.
    - test_accuracies : list of floats
        Accuracy scores on the test sets for each MOFA model.
    - test_f1_scores : list of floats
        Macro F1 scores on the test sets.
    - test_reports : list of pandas DataFrames
        Classification reports (precision, recall, F1) on the test sets.
    - log_models : list of sklearn.linear_model.LogisticRegression
        Trained logistic regression models for each fold.

    Note:
    - Factors are recomputed for both train and test sets to avoid bias due to using factors
      obtained during MOFA training directly.
    - NaN values in omics data are handled by solving the projection equation per sample, 
      mimicking MOFA's handling of missing data without imputation.
    """

    # Lists to store training metrics
    train_accuracies = []
    train_f1_scores = []
    train_reports = []

    # Lists to store test metrics
    test_accuracies = []
    test_f1_scores = []
    test_reports = []
    
    #save logistic regression models from each fold 
    log_model =[]

    # Loop over each MOFA model and its corresponding test fold
    for m, k in zip(model_list, range(len(model_list))):
        
        # Select the test set for each omic
        mRNA_test, DNAm_test, RPPA_test = mRNA_exprs[k], DNAm_exprs[k], RPPA_exprs[k]
        
        # Construct the training set by concatenating all folds except the test fold
        mRNA_train = pd.concat([mRNA_exprs[i] for i in range(len(mRNA_exprs)) if i != k])
        DNAm_train = pd.concat([DNAm_exprs[i] for i in range(len(DNAm_exprs)) if i != k])
        RPPA_train = pd.concat([RPPA_exprs[i] for i in range(len(RPPA_exprs)) if i != k])
    
        # Combine all omics features for training and test sets
        Y_train = pd.concat([mRNA_train, DNAm_train, RPPA_train], axis=1)
        Y_test = pd.concat([mRNA_test, DNAm_test, RPPA_test], axis=1)
        
        # Extract weight matrix W from the MOFA model
        W = m.get_weights(df=True)
    
        # Keep only features used to compute the factors in MOFA model
        Y_train = Y_train[W.index].T
        Y_test = Y_test[W.index].T

        # Convert matrices to numpy arrays for computation
        W_mat = W.values.astype(float) 
        Y_mat_train = Y_train.values.astype(float)  
        n_factors = W_mat.shape[1]
        n_samples = Y_mat_train.shape[1]
    
        # Initialize matrix to store factors for training set
        Z_mat_train = np.zeros((n_factors, n_samples))
        
        # Solve Y = W * Z for each patient individually to handle NaNs
        for i in range(n_samples):
            obs_idx = ~np.isnan(Y_mat_train[:, i])  # indices of non-NaN features
            W_obs = W_mat[obs_idx, :]
            Y_obs = Y_mat_train[obs_idx, i]
            
            if W_obs.shape[0] > 0:
                # Solve least squares problem y = W * z
                Z_mat_train[:, i], *_ = np.linalg.lstsq(W_obs, Y_obs, rcond=None)
            else:
                Z_mat_train[:, i] = np.nan  # if all features are NaN
    
        # Convert factors to DataFrame
        Z_train = pd.DataFrame(Z_mat_train, index=W.columns, columns=Y_train.columns).T
    
        # Keep only factors significantly correlated with tumor subtypes
        Z_train = Z_train[['Factor1', "Factor2", 'Factor7', 'Factor8', 'Factor10', 'Factor11']]
    
        # Extract output labels for training
        y_train = df_all_meta.loc[Z_train.index, 'paper_BRCA_Subtype_PAM50']
    
        # Train logistic regression model
        clf = LogisticRegression(solver='saga',random_state=42, max_iter=10000, n_jobs=1, class_weight='balanced')
        clf.fit(Z_train, y_train)
        y_pred_train = clf.predict(Z_train)
        
        #save fitted model
        log_model.append(clf)
    
        # Compute training accuracy and F1 score
        accuracy = accuracy_score(y_train, y_pred_train)
        f1 = f1_score(y_train, y_pred_train, average='macro')
        train_accuracies.append(accuracy)
        train_f1_scores.append(f1)
    
        # Generate classification report for training set
        report_dict = classification_report(y_train, y_pred_train, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        train_reports.append(df_report)

        # Repeat factor computation and logistic regression for the test set
        Y_mat_test = Y_test.values.astype(float)  
        n_factors = W_mat.shape[1]
        n_samples = Y_mat_test.shape[1]
        Z_mat_test = np.zeros((n_factors, n_samples))

        for i in range(n_samples):
            obs_idx = ~np.isnan(Y_mat_test[:, i])
            W_obs = W_mat[obs_idx, :]         
            Y_obs = Y_mat_test[obs_idx, i]
        
            if W_obs.shape[0] > 0:
                Z_mat_test[:, i], *_ = np.linalg.lstsq(W_obs, Y_obs, rcond=None)
            else:
                Z_mat_test[:, i] = np.nan          

        # Convert test factors to DataFrame and keep selected factors
        Z_test = pd.DataFrame(Z_mat_test, index=W.columns, columns=Y_test.columns).T
        Z_test = Z_test[['Factor1', "Factor2", 'Factor7', 'Factor8', 'Factor10', 'Factor11']]
        y_test = df_all_meta.loc[Z_test.index, 'paper_BRCA_Subtype_PAM50']
   
        # Predict on test set and compute metrics
        y_pred_test = clf.predict(Z_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='macro')
        test_accuracies.append(accuracy)
        test_f1_scores.append(f1)
    
        # Generate classification report for test set
        report_dict = classification_report(y_test, y_pred_test, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        test_reports.append(df_report)
    
    return train_accuracies, train_f1_scores, train_reports, test_accuracies, test_f1_scores, test_reports,log_model


