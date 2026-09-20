from sklearn.model_selection import StratifiedKFold
import pandas as pd

def create_stratified_folds(df, stratify_col, n_splits=5, shuffle=True, random_state=42,
                            save=False, prefix="fold"):
    """
    Create stratified folds of equal size using StratifiedKFold.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to split.
    stratify_col : str
        Column name to use for stratification.
    n_splits : int
        Number of folds (default 5).
    shuffle : bool
        Shuffle before splitting.
    random_state : int
        Random seed for reproducibility.
    save : bool
        Whether to save each fold as a CSV file.
    prefix : str
        Prefix for saved fold filenames (e.g., 'fold_1.csv').

    Returns
    -------
    list of pandas.DataFrame
        A list containing the folds (each ~20% of data).
    """

    X = df
    y = df[stratify_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds = []
    for i, (_, test_idx) in enumerate(skf.split(X, y), 1):
        fold = X.iloc[test_idx].copy()
        folds.append(fold)

        print(f"Fold {i} (size={len(fold)}):")
        print(fold[stratify_col].value_counts(normalize=True))
        print("-" * 40)

        if save:
            fold.to_csv(f"{prefix}_{i}.csv", index=False)

    return folds

def create_subsets_from_fold(df_all_meta,
                             mRNA_expr, DNAm_expr, RPPA_expr,
                             folds):
    """
    Create omics subsets (metadata + expression matrices) based on stratified folds.

    Parameters
    ----------
    df_all_meta : pd.DataFrame
        Metadata indexed by patient ID.
    mRNA_expr, DNAm_expr, RPPA_expr : pd.DataFrame
        Expression matrices indexed by patient ID.
    folds : list of pd.DataFrame
        List of folds created by create_stratified_folds(),
        each containing a 'patient' column.

    Returns
    -------
    lists of dataframes :

         mRNA_exprs, DNAm_exprs, RPPA_exprs,
         all_metas
    """
    # Lists that will store the subsets for each fold
    mRNA_exprs = []
    DNAm_exprs = []
    RPPA_exprs = []

    all_metas = []

    # Iterate through each fold created previously
    for df_sample in folds:

        # Extract the patient IDs contained in the current fold
        patients = df_sample["patient"]

        # --- METADATA ---
        # Select metadata rows whose index matches the patient IDs in this fold
        # 'intersection' avoids errors if some patients do not exist in metadata
        k_meta = df_all_meta.loc[df_all_meta.index.intersection(patients)]

        # --- EXPRESSION MATRICES ---
        # Same selection for each omics matrix (mRNA, DNAm, RPPA)
        k_mRNA = mRNA_expr.loc[mRNA_expr.index.intersection(patients)]
        k_DNAm = DNAm_expr.loc[DNAm_expr.index.intersection(patients)]
        k_RPPA = RPPA_expr.loc[RPPA_expr.index.intersection(patients)]

        # Append the subset of each omics layer to the corresponding list
        mRNA_exprs.append(k_mRNA)
        DNAm_exprs.append(k_DNAm)
        RPPA_exprs.append(k_RPPA)

        # Append the metadata subset
        all_metas.append(k_meta)


    # Return lists of omics subsets (one element per fold) and associated meta data
    return  mRNA_exprs, DNAm_exprs, RPPA_exprs, all_metas


def construct_sumary_dataframe(accuracies, f1_scores, reports,method,name):
    """
    Construct a summary DataFrame that aggregates performance metrics from
    multiple cross-validation folds.


    Parameters
    ----------
    accuracies : list of float
        Accuracy values for each fold.
    f1_scores : list of float
        Macro-averaged F1 scores for each fold.
    reports : list of pandas.DataFrame
        Classification reports (as DataFrames), each containing F1-scores
        for the classes 'Basal', 'Her2', 'LumA', and 'LumB'.
    method:str indicating the imputation method that was used

    Returns
    -------
    pandas.DataFrame
        A transposed summary DataFrame where:
            - Columns = folds
            - Rows = performance metrics
        Plus two additional metadata columns:
            - 'model'
            - 'missing_value_imputation_method'
    """

    dfs = []

    # Iterate over all accuracy, F1-score, and report tuples
    for k, (acc, f1, rep) in enumerate(zip(accuracies, f1_scores, reports)):

        # One row of metrics for this iteration
        data = {
            "Accuracy": acc,
            "Macro F1": f1,
            "Basal F1": rep.loc['Basal', 'f1-score'],
            "Her2 F1": rep.loc['Her2', 'f1-score'],
            "LumA F1": rep.loc['LumA', 'f1-score'],
            "LumB F1": rep.loc['LumB', 'f1-score'],
        }

        # Single-row DataFrame with iteration index
        df_k = pd.DataFrame([data], index=[k])

        # Store it so we can stack all iterations later
        dfs.append(df_k)

    # Concatenate all rows (stack vertically)
    df_final = pd.concat(dfs, axis=0)

    # Optional: transpose the table (rows become columns)
    df_final = df_final.T

    #Add columns with means and std computed over folds
    df_final['mean'] = df_final[[0,1,2,3,4]].mean(axis=1)
    df_final['std'] = df_final[[0,1,2,3,4]].std(axis=1)

    # Add constant columns
    df_final["model"] = name
    df_final["missing_value_imputation_method"] = method


    return df_final