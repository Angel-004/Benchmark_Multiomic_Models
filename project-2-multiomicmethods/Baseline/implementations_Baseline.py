import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

data_dir = "./../data/TCGA-BRCA/"
mod = ["mRNA", "DNAm" , "RPPA"]

class RenameFeatures(BaseEstimator, TransformerMixin):
    '''
    A transformer that renames features to a standard format: PREFIX1, PREFIX2, ..., PREFIXn
    by default, PREFIX is "COMP"
    '''

    def __init__(self, prefix="COMP"):
        self.prefix = prefix
        
    def fit(self, X, y=None):
        # Create new column names
        self.columns_ = [f"{self.prefix}{i+1}" for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        # Preserve index if present
        idx = getattr(X, "index", None)
        # Convert to DataFrame if not already
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            X_transformed.columns = self.columns_
        else:
            X_transformed = pd.DataFrame(np.array(X), columns=self.columns_, index=idx)
        return X_transformed

def merge_omics(data):
    '''
    Merge omic data into a single dataframe.
     Parameters
    - data: dictionary whose keys are those in the universal variable mod, and whose terms contain omic data and metadata
    Returns
    - merged_df: DataFrame containing merged omic data
    - y_full: Series containing the merged labels
    '''
    merged_df = None
    for omic in mod:
        df = data[omic]['expr']
        df = df.add_prefix(f"{omic}_") # to keep track of the omic
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how='outer') # keep all patients and add nans

    print("Merged data shape:", merged_df.shape)
    print(merged_df.head())

    # Merge of the labels
    y_dicts = []
    for omic in mod:
        y_series = data[omic]['meta']['paper_BRCA_Subtype_PAM50']
        y_dicts.append(y_series)

    y_combined = pd.concat(y_dicts, axis=0)

    y_combined = y_combined[~y_combined.index.duplicated()] # Remove duplicates

    y_full = y_combined.reindex(merged_df.index)

    print("Number of labels:", y_full.shape[0])
    print(y_full.head())

    return merged_df, y_full


   

def log_reg(X_train, X_test, y_train, y_test, preprocessings, imputer_included = True, print_report=True, max_iter=5000, random_state=None):
    """
    Preprocessing + logistic regression over a train/test split. Optionally, it drops rows with NaNs from both the training and test set, and prints a classification report.
    
    Parameters
    - X_train, X_test: DataFrames with columns prefixed by omic name (e.g. 'mRNA_')
    - y_train, y_test: pandas Series of labels aligned to X_train/X_test indices
    - preprocessings: a dictionary of sklearn transformer pipelines or transformers that support fit_transform/transform. It contains preprocessing steps to apply per omic. It must be keyed by omic name ('mRNA', 'DNAm', 'RPPA')
    - imputer_included: if False, rows with NA values will be dropped from both train and test sets by default
    - print_report: if True, prints a classification report
    - max_iter: maximum number of iterations for the logistic regression
    - random_state: random seed for reproducibility
    
    Returns
    - y_pred, accuracy, macro_f1: predicted labels for X_test, accuracy score and macro F1 score
    - logistic: trained logistic regression model
    """


    if not imputer_included:
        # Drop samples with missing values
        print("No imputation method added. Dropping samples with missing values.")
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]

    X_train, X_test = apply_per_omic(X_train, X_test, preprocessings)

    logistic = LogisticRegression(
        solver="saga",
        max_iter=max_iter,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state
    )
    logistic.fit(X_train, y_train)

    y_pred = logistic.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    if print_report:
        print(classification_report(y_test, y_pred)) 
    return y_pred, accuracy, macro_f1, logistic


def crossval_log_reg(X, y, preprocessings, imputer_included = True, print_report=True, return_full_data = False, n_splits=5, max_iter=5000, random_state=None):
    """
    crossvalidation over a preprocessing + logistic regression pipeline over a train/test split. Optionally, it drops rows with NaNs from both the training and test set, and prints a classification report.
    
    Parameters
    - X: DataFrame with columns prefixed by omic name (e.g. 'mRNA_')
    - y: pandas Series of labels aligned to the indices of X
    - preprocessings: a dictionary of sklearn transformer pipelines or transformers that support fit_transform/transform. It contains preprocessing steps to apply per omic. It must be keyed by omic name ('mRNA', 'DNAm', 'RPPA')
    - imputer_included: if False, rows with NA values will be dropped from both train and test sets by default
    - print_report: if True, prints a classification report
    - return_full_data: if True, returns the full data including the trained models of each fold and the corresponding splitted training data indices
    - n_splits: number of folds for the StratifiedKFold cross-validation
    - max_iter: maximum number of iterations for the logistic regression
    - random_state: random seed for reproducibility
    
    Returns
    - accuracies, macro_f1s: predicted labels for X_test, accuracy score and macro F1 score
    - (if return_full_data) models, splits_train: list of trained logistic regression models per fold and list of corresponding training data indices 
    """

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracies = []
    macro_f1s = []
    models = []
    splits_train = []

    fold = 1
    for train_idx, test_idx in kf.split(X, y):

        print(f"\n========== Fold {fold} ==========")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        y_pred, accuracy, macro_f1, logistic = log_reg(X_train, X_test, y_train, y_test, preprocessings, imputer_included=imputer_included, print_report=print_report, max_iter=max_iter, random_state=random_state)
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)
        models.append(logistic)
        splits_train.append(train_idx)        
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        print(f"Fold {fold} macro F1: {macro_f1:.4f}")
        fold += 1
    
    print("")
    print("Final results:")
    print("")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"Best macro F1 was obtained at fold {np.argmax(macro_f1s)+1} with value {np.max(macro_f1s):.4f}")
    if return_full_data:
        return accuracies, macro_f1s, models, splits_train
    else:
        return  accuracies, macro_f1s



def get_omic_split(X, omic_name):
    '''
    Extracts the columns corresponding to a specific omic from the merged DataFrame. 

    Parameters
    - X: DataFrame with columns prefixed by omic name (e.g. 'mRNA_')
    - omic_name: string indicating the omic to extract (e.g. 'mRNA')
    
    Returns
    - omic_X: DataFrame containing only the columns of the specified omic
    '''
    return X[[c for c in X.columns if c.startswith(f"{omic_name}_")]]


def apply_per_omic(X_train, X_test, pipelines):
    '''
    Applies a pipeline on each omic separately and combines the results.

    Parameters
    - X_train, X_test: Training and test dataframes with columns prefixed by omic name (e.g. 'mRNA_')
    - pipelines: a dictionary of sklearn transformer pipelines or transformers that support fit_transform/transform. It contains preprocessing steps to apply per omic. It must be keyed by omic name ('mRNA', 'DNAm', 'RPPA')
    
    Returns
    - X_train_final, X_test_final: transformed DataFrames
    '''
    # Sanity check
    if not isinstance(pipelines, dict) or set(mod) != set(pipelines.keys()):
        got = sorted(pipelines.keys()) if isinstance(pipelines, dict) else str(type(pipelines))
        raise ValueError(f"apply_per_omic: 'pipelines' must be a dict with keys exactly {sorted(mod)}. Got: {got}")


    X_train_parts = []
    X_test_parts = []
    cols_per_omic = []

    for omic in mod:
        X_train_part = get_omic_split(X_train, omic)
        X_test_part = get_omic_split(X_test, omic)

        # clone the pipeline so each omic gets its own fitted steps
        pipe = clone(pipelines[omic])
 
        X_train_trans = pipe.fit_transform(X_train_part)
        X_test_trans = pipe.transform(X_test_part)

        # ensure 2D
        if getattr(X_train_trans, 'ndim', 1) != 2 or getattr(X_test_trans, 'ndim', 1) != 2:
            raise ValueError("Pipeline must output 2D arrays (n_samples, n_components).")

        #n_comp = X_train_trans.shape[1]
        #cols = [f"{omic}_COMP{i+1}" for i in range(n_comp)] #might have to change this for the interpretability part
        #cols_per_omic.append(cols)

        X_train_parts.append(X_train_trans)
        X_test_parts.append(X_test_trans)

    X_train_final = pd.concat(X_train_parts, axis=1)
    X_test_final = pd.concat(X_test_parts, axis=1)


    return X_train_final, X_test_final


def loadings_heatmap(var_expl, omic_subdata, omic_data, omic, n = 1, top_k_heat=10):

    '''
    Plots a heatmap and a table of the top_k_heat features with highest absolute loadings for PC n of the specified omic, using the dataframe omic_subdata.
    PCA is performed so that var_expl is the percentage of variance explained.

    Parameters
    - var_expl: float between 0 and 1 indicating the percentage of variance to be explained by PCA
    - omic_data: dictionary whose keys are those in the universal variable mod, and whose terms contain omic data and metadata
    - omic_subdata: dictionary whose keys are those in the universal variable mod, so that omic_subdata[omic] contains omic data for the specified omic
    - omic: string indicating the omic to analyze. It must be one of those in mod.
    - n: integer indicating the principal component to analyze (1-based index)
    - top_k_heat: integer indicating how many top features to show in the heatmap
    
    Returns
    - None (it plots the heatmap and prints the table)
    '''
    if omic not in mod:
        raise ValueError(f"omic must be one of {mod}. Got: {omic}")

    X = omic_subdata[omic]
    pca = PCA(n_components=var_expl)
    X_pca = pca.fit_transform(X)
    X_pca = StandardScaler().fit_transform(X_pca)

    pcs = pd.DataFrame(X_pca, index=X.index,
                   columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    
    pcs['paper_BRCA_Subtype_PAM50'] = omic_data[omic]['meta'].loc[pcs.index, 'paper_BRCA_Subtype_PAM50']


    pc_idx = n - 1

    if pc_idx > pca.components_.shape[0]:
        raise ValueError(f"PC {n} not available: PCA has only {pca.components_.shape[0]} components")

    pc_loadings_n = pca.components_[pc_idx]
    df_load_n = pd.DataFrame({
        "feature": X.columns,
        "loading": pc_loadings_n
    })
    # order by absolute loading but keep signed values
    df_load_n["abs_loading"] = df_load_n["loading"].abs()
    df_load_n = df_load_n.sort_values("abs_loading", ascending=False).reset_index(drop=True)

    # choose how many top features to show in the heatmap
    df_heat = df_load_n.head(top_k_heat).set_index("feature")[["loading"]]

    plt.figure(figsize=(8, 2 * max(2, 0.25 * len(df_heat))))
    sns.heatmap(df_heat, cmap="coolwarm", center=0, annot=True, fmt=".3f", cbar_kws={"label": "Loading"})
    plt.title(f"{omic} loadings for PC{n} (top {len(df_heat)} by |loading|)")
    plt.ylabel("Feature")
    plt.xlabel("Loading")
    plt.tight_layout()
    plt.show()

    #Table with top features
    print(f"Top {top_k_heat} {omic} features by absolute loading on PC{n}:")
    print(df_heat)
