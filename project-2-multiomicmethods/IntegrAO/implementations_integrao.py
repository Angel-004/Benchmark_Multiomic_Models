import os
import sys

NOTEBOOK_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utility import *
from integrao.integrater import integrao_integrater, integrao_predictor
from integrao.dataset import GraphDataset
from integrao.unsupervised_train import tsne_p_deep
from integrao.supervised_train import tsne_p_deep_classification
from integrao.main import dist2, integrao_fuse, _stable_normalized
from integrao.util import data_indexing

import snf

def fold_preprocess_integrao(fold, omic_matrices, top_var_nbr=2000):
  """
    Preprocess a fold for integrAO training.

    Parameters
    ----------
    fold : pd.DataFrame or index
        Fold test indices.
    omic_matrices : dict
        Dictionary of omics DataFrames keyed by modality.
    top_var_nbr : int, optional
        Number of top variable features to keep per modality (default 2000).

    Returns
    -------
    mRNA_ref, DNAm_ref, RPPA_ref : pd.DataFrame
        Training/reference matrices for each modality.
    mRNA_query, DNAm_query, RPPA_query : pd.DataFrame
        Test/query matrices for each modality.
    X_train : list
        Training sample IDs.
    X_test : list
        Test sample IDs.
    """
  patients = set(omic_matrices["mRNA"].index)

  X_test = fold.index
  X_train = [x for x in patients if x not in X_test]


  mRNA_ref = omic_matrices["mRNA"].loc[X_train]
  DNAm_ref = omic_matrices["DNAm"].loc[X_train]
  RPPA_ref = omic_matrices["RPPA"].loc[X_train]

  mRNA_query = omic_matrices["mRNA"].loc[X_test]
  DNAm_query = omic_matrices["DNAm"].loc[X_test]
  RPPA_query = omic_matrices["RPPA"].loc[X_test]

  mRNA_top = mRNA_ref.var().sort_values(ascending=False).index[:min(top_var_nbr, mRNA_ref.shape[1])]
  DNAm_top = DNAm_ref.var().sort_values(ascending=False).index[:min(top_var_nbr, DNAm_ref.shape[1])]
  RPPA_top = RPPA_ref.var().sort_values(ascending=False).index[:min(top_var_nbr, RPPA_ref.shape[1])]

  mRNA_ref = mRNA_ref[mRNA_top].dropna()
  DNAm_ref = DNAm_ref[DNAm_top].dropna()
  RPPA_ref = RPPA_ref[RPPA_top].dropna()

  mRNA_query = mRNA_query[mRNA_top].dropna()
  DNAm_query = DNAm_query[DNAm_top].dropna()
  RPPA_query = RPPA_query[RPPA_top].dropna()

  '''mRNA_scaler = StandardScaler()
  DNAm_scaler = StandardScaler()
  RPPA_scaler = StandardScaler()

  mRNA_ref.iloc[:] = mRNA_scaler.fit_transform(mRNA_ref)
  mRNA_query.iloc[:] = mRNA_scaler.transform(mRNA_query)

  DNAm_ref.iloc[:] = DNAm_scaler.fit_transform(DNAm_ref)
  DNAm_query.iloc[:] = DNAm_scaler.transform(DNAm_query)

  RPPA_ref.iloc[:] = RPPA_scaler.fit_transform(RPPA_ref)
  RPPA_query.iloc[:] = RPPA_scaler.transform(RPPA_query)'''

  return mRNA_ref, DNAm_ref, RPPA_ref, mRNA_query, DNAm_query, RPPA_query, X_train, X_test

def get_metrics(preds, preds_index, X_test, y_test):
    """
    Compute evaluation metrics for predicted labels.

    Parameters
    ----------
    preds : array-like or DataFrame
        Predicted labels for all samples.
    preds_index : list
        Index of samples corresponding to predictions.
    X_test : list
        List of test sample IDs to evaluate.
    y_test : array-like
        True labels for test samples.

    Returns
    -------
    f1_macro : float
        Macro-averaged F1 score.
    f1_weighted : float
        Weighted-averaged F1 score.
    acc : float
        Overall accuracy.
    f1_per_class : np.ndarray
        F1 score per class.
    accuracy_per_class : np.ndarray
        Accuracy per class.
    """
    pred_df = pd.DataFrame(data=preds, index=preds_index)
    pred_df_test = pred_df.loc[X_test]

    pred_df_test_values = pred_df_test.values.flatten().astype(int)
    y_test_values = np.array(y_test).astype(int)

    f1_micro = f1_score(y_test_values, pred_df_test_values, average='macro')
    f1_weighted = f1_score(y_test_values, pred_df_test_values, average='weighted')
    acc = accuracy_score(y_test_values, pred_df_test_values)

    # Per-class metrics
    f1_per_class = f1_score(y_test_values, pred_df_test_values, average=None)
    accuracy_per_class = []
    for class_label in np.unique(y_test_values):
        class_mask = y_test_values == class_label
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test_values[class_mask], pred_df_test_values[class_mask])
            accuracy_per_class.append(class_acc)
        else:
            accuracy_per_class.append(np.nan)
    accuracy_per_class = np.array(accuracy_per_class)

    return f1_micro, f1_weighted, acc, f1_per_class, accuracy_per_class

def display_metrics(f1_micro, f1_weighted, acc, f1_per_class, accuracy_per_class, class_names=None, fold_num=None):
    """
    Cleanly display all metrics including micro, weighted, and per-class metrics.

    Parameters
    ----------
    f1_micro : float
        Micro-averaged F1 score
    f1_weighted : float
        Weighted-averaged F1 score
    acc : float
        Overall accuracy
    f1_per_class : array-like
        Per-class F1 scores
    accuracy_per_class : array-like
        Per-class accuracies
    class_names : list, optional
        Names of the classes (if None, uses class indices)
    fold_num : int, optional
        Fold number for display purposes
    """

    header = f"Fold {fold_num}" if fold_num is not None else "Metrics"
    print("\n" + "="*70)
    print(f"  {header}".ljust(70))
    print("="*70)

    print("\nOverall Metrics:")
    print(f"  F1 (macro):    {f1_micro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  Accuracy:      {acc:.4f}")

    print("\nPer-Class Metrics:")
    print("-" * 70)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(f1_per_class))]

    print(f"{'Class':<20} {'F1 Score':<20} {'Accuracy':<20}")
    print("-" * 70)
    for i, (f1, class_acc) in enumerate(zip(f1_per_class, accuracy_per_class)):
        class_label = class_names[i] if i < len(class_names) else f"Class {i}"
        f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
        acc_str = f"{class_acc:.4f}" if not np.isnan(class_acc) else "N/A"
        print(f"{class_label:<20} {f1_str:<20} {acc_str:<20}")

    print("="*70)

def display_cv_summary(fold_results, class_names=None):
    """
    Display cross-validation summary with mean and std across all folds.

    Parameters
    ----------
    fold_results : list of dict
        List of fold result dictionaries from training_integrao_crossval
    class_names : list, optional
        Names of the classes (if None, uses class indices)
    """

    # Extract metrics
    f1_micro_list = [r['f1_micro'] for r in fold_results]
    f1_weighted_list = [r['f1_weighted'] for r in fold_results]
    acc_list = [r['acc'] for r in fold_results]

    # Aggregate per-class metrics across folds
    num_classes = len(fold_results[0]['f1_per_class'])
    f1_per_class_list = [[] for _ in range(num_classes)]
    accuracy_per_class_list = [[] for _ in range(num_classes)]

    for result in fold_results:
        for class_idx in range(num_classes):
            f1_per_class_list[class_idx].append(result['f1_per_class'][class_idx])
            accuracy_per_class_list[class_idx].append(result['accuracy_per_class'][class_idx])

    # Compute mean and std
    f1_per_class_mean = [np.nanmean(f1s) for f1s in f1_per_class_list]
    f1_per_class_std = [np.nanstd(f1s) for f1s in f1_per_class_list]
    accuracy_per_class_mean = [np.nanmean(accs) for accs in accuracy_per_class_list]
    accuracy_per_class_std = [np.nanstd(accs) for accs in accuracy_per_class_list]

    # Display
    print("\n" + "="*90)
    print("  CROSS-VALIDATION SUMMARY (Mean ± Std across all folds)")
    print("="*90)

    print("\nOverall Metrics:")
    print(f"  F1 (micro):    {np.mean(f1_micro_list):.4f} ± {np.std(f1_micro_list):.4f}")
    print(f"  F1 (weighted): {np.mean(f1_weighted_list):.4f} ± {np.std(f1_weighted_list):.4f}")
    print(f"  Accuracy:      {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")

    print("\nPer-Class Metrics (Mean ± Std):")
    print("-" * 90)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    print(f"{'Class':<20} {'F1 Score':<35} {'Accuracy':<35}")
    print("-" * 90)
    for i in range(num_classes):
        class_label = class_names[i] if i < len(class_names) else f"Class {i}"
        f1_str = f"{f1_per_class_mean[i]:.4f} ± {f1_per_class_std[i]:.4f}"
        acc_str = f"{accuracy_per_class_mean[i]:.4f} ± {accuracy_per_class_std[i]:.4f}"
        print(f"{class_label:<20} {f1_str:<35} {acc_str:<35}")

    print("="*90)

def training_integrao_crossval(folds, omic_matrices, model_path, top_var_nbr, dataset_name, neighbor_size, embedding_dims,
                               fusing_iteration, normalization_factor, alighment_epochs, beta, mu, y, result_dir, finetune_epochs, cluster_number,
                               modalities_name_list=["mRNA", "DNAm", "RPPA"]):
  """
    Train integrAO with cross-validation.

    Parameters
    ----------
    folds : list
        List of fold DataFrames or indices.
    omic_matrices : dict
        Dictionary of omics DataFrames keyed by modality.
    model_path : str
        Path to save/load models.
    top_var_nbr : int
        Number of top variable features to select per modality.
    dataset_name : str
        Name of the dataset.
    neighbor_size : int
        Number of neighbors for similarity network construction.
    embedding_dims : int
        Dimension of latent embeddings.
    fusing_iteration : int
        Number of iterations for network fusion.
    normalization_factor : float
        Scaling factor for normalized similarity matrices.
    alighment_epochs : int
        Number of training epochs for embedding alignment.
    beta : float
        SNF parameter for affinity calculation.
    mu : float
        SNF parameter for affinity calculation.
    y : pd.Series or DataFrame
        Ground truth labels.
    result_dir : str
        Directory to save models and results.
    finetune_epochs : int
        Number of epochs for supervised fine-tuning.
    cluster_number : int
        Number of output classes/clusters.
    modalities_name_list : list of str, optional
        Names of omics modalities (default ["mRNA", "DNAm", "RPPA"]).

    Returns
    -------
    fold_results : list of dict
        Metrics for each fold including per-class scores.
    best_queries : tuple
        Query datasets (mRNA, DNAm, RPPA) from the best-performing fold.
    """
  count = 1

  #metrics
  f1_micro_list = []
  f1_weight_list = []
  acc_list = []
  fold_results = []
  models_list = []  # List to store all trained models
  queries_list = []  # List to store query datasets for each fold

  for fold in folds:
    print(f"\n{'='*70}")
    print(f"Processing Fold {count}/{len(folds)}")
    print(f"{'='*70}")

    mRNA_ref, DNAm_ref, RPPA_ref, mRNA_query, DNAm_query, RPPA_query, X_train, X_test = fold_preprocess_integrao(fold, omic_matrices, top_var_nbr=top_var_nbr)

    queries_list.append((mRNA_query, DNAm_query, RPPA_query))

    ## -- Integrates reference data --

    integrater = integrao_integrater(
      [mRNA_ref, DNAm_ref, RPPA_ref],
      dataset_name,
      modalities_name_list=["mRNA", "DNAm", "RPPA"],   # used for naming the incomplete modalities during new sample inference
      neighbor_size=neighbor_size,
      embedding_dims=embedding_dims,
      fusing_iteration=fusing_iteration,
      normalization_factor=normalization_factor,
      alighment_epochs=alighment_epochs,
      beta=beta,
      mu=mu,
    )

    # data indexing
    fused_networks = integrater.network_diffusion()
    embeds_final, S_final, model = integrater.unsupervised_alignment()

    model = model.to(device)

    # save the model for fine-tuning
    torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))


    ## -- Fine-tuning --

    # Converts y to a Series
    if isinstance(y, pd.DataFrame):
      if y.shape[1] == 1:
        y_series = y.iloc[:, 0]
      else:
        y_series = y
    else:
      y_series = y

    label_map = {label: i for i, label in enumerate(sorted([label for label in y_series.unique() if pd.notna(label)]))}
    truelabel_sub = pd.DataFrame({'subjects': embeds_final.index,'cluster.id': y_series.loc[embeds_final.index].map(label_map)})
    truelabel_sub = truelabel_sub.set_index('subjects')

    embeds_final, S_final, model, preds = integrater.classification_finetuning(truelabel_sub, result_dir, finetune_epochs=finetune_epochs)

    model = model.to(device)

    # Store the model in the list
    models_list.append(model)

    # Saving
    torch.save(model.state_dict(), os.path.join(result_dir, "model_integrao_supervised.pth"))

    ## -- Prediction --

    # Setup
    predictor = integrao_predictor(
      [mRNA_query, DNAm_query, RPPA_query],
      dataset_name,
      modalities_name_list=["mRNA", "DNAm", "RPPA"],
      neighbor_size=neighbor_size,
      embedding_dims=embedding_dims,
      fusing_iteration=fusing_iteration,
      normalization_factor=normalization_factor,
      alighment_epochs=alighment_epochs,
      beta=beta,
      mu=mu,
      num_classes=cluster_number
    )

    fused_networks = predictor.network_diffusion()

    preds = predictor.inference_supervised(
      model_path,
      new_datasets=[mRNA_query, DNAm_query, RPPA_query],
      modalities_names=["mRNA", "DNAm", "RPPA"]
    )

    preds_index = []
    seen_samples = set()
    for df, name in zip([mRNA_query, DNAm_query, RPPA_query], ["mRNA", "DNAm", "RPPA"]):
        domain_index = predictor.modalities_name_list.index(name)
        domain_samples = predictor.dict_original_order[domain_index]
        for sample_id in domain_samples:
            if sample_id in df.index and sample_id not in seen_samples:
                preds_index.append(sample_id)
                seen_samples.add(sample_id)

    labels_series = y_series.reindex(preds_index).map(label_map)
    valid_mask = labels_series.notna()
    if valid_mask.sum() == 0:
        raise ValueError("No valid labels for the predicted samples")

    valid_samples = labels_series[valid_mask].index
    y_test_int = labels_series[valid_mask].astype(int).tolist()

    preds_filtered = pd.DataFrame(preds, index=preds_index).loc[valid_samples]

    X_test_filtered = [sid for sid in preds_filtered.index if sid in X_test]

    # Sanity check
    assert len(preds_filtered) == len(y_test_int) == len(X_test_filtered), "Alignment error!"

    f1_micro, f1_weight, acc, f1_per_class, accuracy_per_class = get_metrics(
        preds_filtered,
        preds_filtered.index,
        X_test_filtered,
        y_test_int
    )

    f1_micro_list.append(f1_micro)
    f1_weight_list.append(f1_weight)
    acc_list.append(acc)

    # Store fold results with per-class metrics
    fold_results.append({
        'fold': count,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weight,
        'acc': acc,
        'f1_per_class': f1_per_class,
        'accuracy_per_class': accuracy_per_class
    })

    # Create class names from label map (reverse mapping)
    class_names = [None] * len(label_map)
    for label, idx in label_map.items():
        class_names[idx] = str(label)

    # Display metrics for this fold
    display_metrics(f1_micro, f1_weight, acc, f1_per_class, accuracy_per_class,
                   class_names=class_names, fold_num=count)

    count = count + 1

  print("\n" + "="*70)
  print("  CROSS-VALIDATION SUMMARY")
  print("="*70)
  print(f"F1 macro:   {np.mean(f1_micro_list):.4f} ± {np.std(f1_micro_list):.4f}")
  print(f"F1 weighted: {np.mean(f1_weight_list):.4f} ± {np.std(f1_weight_list):.4f}")
  print(f"Accuracy:   {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
  print("="*70)

  # Find the best model based on F1 macro score
  best_fold_idx = np.argmax(f1_micro_list)
  best_model = models_list[best_fold_idx]
  torch.save(best_model.state_dict(), os.path.join(result_dir, "best_model_integrao_supervised.pth"))
  print(f"Saved the best model from fold {best_fold_idx + 1} with F1 macro: {f1_micro_list[best_fold_idx]:.4f}")

  best_queries = queries_list[best_fold_idx]

  return fold_results, best_queries

class integrao_integrater(object):
    def __init__(
        self,
        datasets,
        dataset_name=None,
        modalities_name_list=None,
        neighbor_size=None,
        embedding_dims=50,
        fusing_iteration=20,
        normalization_factor=1.0,
        alighment_epochs=1000,
        beta=1.0,
        mu=0.5,
        random_state=42,
    ):
        self.datasets = datasets
        self.dataset_name = dataset_name
        self.modalities_name_list = modalities_name_list
        self.embedding_dims = embedding_dims
        self.fusing_iteration = fusing_iteration
        self.normalization_factor = normalization_factor
        self.alighment_epochs = alighment_epochs
        self.beta = beta
        self.mu = mu
        self.random_state=random_state

        # data indexing
        (
            self.dicts_common,
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            self.dicts_unique,
            self.original_order,
            self.dict_original_order,
        ) = data_indexing(self.datasets)

        # set neighbor size
        if neighbor_size == None:
            self.neighbor_size = int(datasets[0].shape[0] / 6)
        else:
            self.neighbor_size = neighbor_size
        print("Neighbor size:", self.neighbor_size)

    def network_diffusion(self):
        S_dfs = []
        for i in range(0, len(self.datasets)):
            view = self.datasets[i]
            dist_mat = dist2(view.values, view.values)
            S_mat = snf.compute.affinity_matrix(
                dist_mat, K=self.neighbor_size, mu=self.mu
            )

            S_df = pd.DataFrame(
                data=S_mat, index=self.original_order[i], columns=self.original_order[i]
            )

            S_dfs.append(S_df)

        self.fused_networks = integrao_fuse(
            S_dfs.copy(),
            dicts_common=self.dicts_common,
            dicts_unique=self.dicts_unique,
            original_order=self.original_order,
            neighbor_size=self.neighbor_size,
            fusing_iteration=self.fusing_iteration,
            normalization_factor=self.normalization_factor,
        )
        return self.fused_networks

    def unsupervised_alignment(self):
        # turn pandas dataframe into np array
        datasets_val = [x.values for x in self.datasets]
        fused_networks_val = [x.values for x in self.fused_networks]

        S_final, self.models = tsne_p_deep(
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            datasets_val,
            P=fused_networks_val,
            neighbor_size=self.neighbor_size,
            embedding_dims=self.embedding_dims,
            alighment_epochs=self.alighment_epochs,
        )

        self.final_embeds = pd.DataFrame(
            data=S_final, index=self.dict_sampleToIndexs.keys()
        )
        self.final_embeds.sort_index(inplace=True)

        # calculate the final similarity graph
        dist_final = dist2(self.final_embeds.values, self.final_embeds.values)
        Wall_final = snf.compute.affinity_matrix(
            dist_final, K=self.neighbor_size, mu=self.mu
        )

        Wall_final = _stable_normalized(Wall_final)

        return self.final_embeds, Wall_final, self.models

    def classification_finetuning(self, clf_labels, model_path, finetune_epochs=1000):
        # turn pandas dataframe into np array
        datasets_val = [x.values for x in self.datasets]
        fused_networks_val = [x.values for x in self.fused_networks]

        # reorder of clf_labels to make it the same with self.dict_sampleToIndexs.keys()
        clf_labels = clf_labels.loc[self.dict_sampleToIndexs.keys()]

        S_final, self.models, preds = tsne_p_deep_classification(
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            self.dict_original_order,
            datasets_val,
            clf_labels,
            P=fused_networks_val,
            model_path=model_path,
            neighbor_size=self.neighbor_size,
            embedding_dims=self.embedding_dims,
            alighment_epochs=finetune_epochs,
            num_classes=len(np.unique(clf_labels)),
        )

        self.final_embeds = pd.DataFrame(
            data=S_final, index=self.dict_sampleToIndexs.keys()
        )
        self.final_embeds.sort_index(inplace=True)

        # calculate the final similarity graph
        dist_final = dist2(self.final_embeds.values, self.final_embeds.values)
        Wall_final = snf.compute.affinity_matrix(
            dist_final, K=self.neighbor_size, mu=self.mu
        )

        Wall_final = _stable_normalized(Wall_final)

        return self.final_embeds, Wall_final, self.models, preds


class integrao_predictor(object):
    def __init__(
        self,
        datasets,
        dataset_name=None,
        modalities_name_list=None,
        neighbor_size=None,
        embedding_dims=50,
        hidden_channels=128,
        fusing_iteration=20,
        normalization_factor=1.0,
        alighment_epochs=1000,
        beta=1.0,
        mu=0.5,
        num_classes=None,
    ):
        self.datasets = datasets
        self.dataset_name = dataset_name
        self.modalities_name_list = modalities_name_list
        self.embedding_dims = embedding_dims
        self.hidden_channels = hidden_channels
        self.fusing_iteration = fusing_iteration
        self.normalization_factor = normalization_factor
        self.alighment_epochs = alighment_epochs
        self.beta = beta
        self.mu = mu
        self.num_classes = num_classes

        # data indexing
        (
            self.dicts_common,
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            self.dicts_unique,
            self.original_order,
            self.dict_original_order,
        ) = data_indexing(self.datasets)

        # set neighbor size
        if neighbor_size == None:
            self.neighbor_size = int(datasets[0].shape[0] / 6)
        else:
            self.neighbor_size = neighbor_size
        print("Neighbor size:", self.neighbor_size)

        self.feature_dims = []
        for i in range(len(self.datasets)):
            self.feature_dims.append(np.shape(self.datasets[i])[1])

        if num_classes is not None:
            self.num_classes = num_classes


    def network_diffusion(self):
        S_dfs = []
        for i in range(0, len(self.datasets)):
            view = self.datasets[i]
            dist_mat = dist2(view.values, view.values)
            S_mat = snf.compute.affinity_matrix(
                dist_mat, K=self.neighbor_size, mu=self.mu
            )

            S_df = pd.DataFrame(
                data=S_mat, index=self.original_order[i], columns=self.original_order[i]
            )

            S_dfs.append(S_df)

        self.fused_networks = integrao_fuse(
            S_dfs.copy(),
            dicts_common=self.dicts_common,
            dicts_unique=self.dicts_unique,
            original_order=self.original_order,
            neighbor_size=self.neighbor_size,
            fusing_iteration=self.fusing_iteration,
            normalization_factor=self.normalization_factor,
        )
        return self.fused_networks


    def _load_pre_trained_weights(self, model, model_path, device):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def inference_unsupervised(self, model_path, new_datasets, modalities_names):
        # loop through the new_dataset and create Graphdatase
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from integrao.IntegrAO_unsupervised import IntegrAO
        model = IntegrAO(self.feature_dims, self.hidden_channels, self.embedding_dims).to(device)
        model = self._load_pre_trained_weights(model, model_path, device)

        x_dict = {}
        edge_index_dict = {}
        for i, modal in enumerate(new_datasets):
            # find the index of the modal in the self.modalities_name_list
            model_name = modalities_names[i]
            modal_index = self.modalities_name_list.index(model_name)

            dataset = GraphDataset(
                self.neighbor_size,
                modal.values,
                self.fused_networks[modal_index].values,
                transform=T.ToDevice(device),
            )
            modal_dg = dataset[0]

            x_dict[modal_index] = modal_dg.x
            edge_index_dict[modal_index] = modal_dg.edge_index

        # Now to do the inference
        # ---------------------------------------------------------
        embeddings= model(x_dict, edge_index_dict)
        for i in range(len(new_datasets)):
            embeddings[i] = embeddings[i].detach().cpu().numpy()

        final_embedding = np.array([]).reshape(0, self.embedding_dims)
        for key in self.dict_sampleToIndexs:
            sample_embedding = np.zeros((1, self.embedding_dims))

            for (dataset, index) in self.dict_sampleToIndexs[key]:
                sample_embedding += embeddings[dataset][index]
            sample_embedding /= len(self.dict_sampleToIndexs[key])

            final_embedding = np.concatenate((final_embedding, sample_embedding), axis=0)

        # Now format the final embeddings
        # ---------------------------------------------------------
        final_embedding_df = pd.DataFrame(
            data=final_embedding, index=self.dict_sampleToIndexs.keys()
        )
        final_embedding_df.sort_index(inplace=True)

        # calculate the final similarity graph
        dist_final = dist2(final_embedding_df.values, final_embedding_df.values)
        Wall_final = snf.compute.affinity_matrix(
            dist_final, K=self.neighbor_size, mu=self.mu
        )

        Wall_final = _stable_normalized(Wall_final)

        return final_embedding_df, Wall_final

    def interpret_unsupervised(self, model_path, result_dir, new_datasets, modalities_names):
        # loop through the new_dataset and create Graphdatase
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from integrao.IntegrAO_unsupervised import IntegrAO
        model = IntegrAO(self.feature_dims, self.hidden_channels, self.embedding_dims).to(device)
        model = self._load_pre_trained_weights(model, model_path, device)


        # explain the model
        from captum.attr import IntegratedGradients

        # It takes as input the variable node features for one domain,
        # while the remaining features and edge indices remain fixed.
        def custom_forward(x, static_x_dict, edge_index_dict, domain):
            x_dict = static_x_dict.copy()
            x_dict[domain] = x

            out_dict = model(x_dict, edge_index_dict)

            return out_dict[domain].sum(dim=1)   # iG requires scalar output; so we sum the output of the embeddings

        # prepare the data
        x_dict = {}
        edge_index_dict = {}
        for i, modal in enumerate(new_datasets):
            model_name = modalities_names[i]
            modal_index = self.modalities_name_list.index(model_name)

            dataset = GraphDataset(
                self.neighbor_size,
                modal.values,
                self.fused_networks[modal_index].values,
                transform=T.ToDevice(device),
            )
            modal_dg = dataset[0]

            x_dict[modal_index] = modal_dg.x
            edge_index_dict[modal_index] = modal_dg.edge_index

        # Loop over each domain (modality)
        # ---------------------------------------------------------
        feat_importances = {}
        for domain in x_dict:
            x_input = x_dict[domain] # The variable input for the current domain.
            static_x = {k: x_dict[k] for k in x_dict}

            ig = IntegratedGradients(custom_forward)

            attributions, delta = ig.attribute(
                inputs=x_input,
                additional_forward_args=(static_x, edge_index_dict, domain),
                return_convergence_delta=True
            )

            if domain not in feat_importances:
                feat_importances[domain] = []
            feat_importances[domain].append(attributions.detach().cpu().numpy())


        df_list = []
        for domain in feat_importances:

            # Concatenate along the first axis (nodes).
            feat_importances[domain] = np.concatenate(feat_importances[domain], axis=0)
            num_feats = feat_importances[domain].shape[1]
            # Create a DataFrame; here columns are named feat_0, feat_1, etc.
            df = pd.DataFrame(feat_importances[domain], columns=[f'feat_{i}' for i in range(num_feats)])
            df_list.append(df)

            # save the feature importance
            csv_path = os.path.join(result_dir, f'{modalities_names[domain]}_feat_importance.csv')
            df.to_csv(csv_path, index=False)
            print(df.shape)

            print(f"Saved feature importances for domain {modalities_names[domain]} to {csv_path}")

        return  df_list


    def inference_supervised(self, model_path, new_datasets, modalities_names):
        # loop through the new_dataset and create Graphdatase
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from integrao.IntegrAO_supervised import IntegrAO
        model = IntegrAO(self.feature_dims, self.hidden_channels, self.embedding_dims, num_classes=self.num_classes).to(device)
        model = self._load_pre_trained_weights(model, model_path, device)

        x_dict = {}
        edge_index_dict = {}
        for i, modal in enumerate(new_datasets):
            # find the index of the modal in the self.modalities_name_list
            model_name = modalities_names[i]
            modal_index = self.modalities_name_list.index(model_name)

            dataset = GraphDataset(
                self.neighbor_size,
                modal.values,
                self.fused_networks[modal_index].values,
                transform=T.ToDevice(device),
            )
            modal_dg = dataset[0]

            x_dict[modal_index] = modal_dg.x
            edge_index_dict[modal_index] = modal_dg.edge_index

        # Now to do the inference
        final_embeddings, _, preds, id_list = model(
            x_dict, edge_index_dict, self.dict_original_order
        )

        preds = F.softmax(preds, dim=1)
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)

        return preds


    def interpret_supervised(self, model_path, result_dir, new_datasets, modalities_names):
        # loop through the new_dataset and create Graphdatase
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from integrao.IntegrAO_supervised import IntegrAO
        model = IntegrAO(self.feature_dims, self.hidden_channels, self.embedding_dims, num_classes=self.num_classes).to(device)
        model = self._load_pre_trained_weights(model, model_path, device)

        # explain the model
        from captum.attr import IntegratedGradients

        # It takes variable node features (x) for a given domain,
        # while keeping the rest of the inputs (static_x_dict, edge_index_dict, and domain_sample_ids) fixed.
        def custom_forward(x_to_attr, static_x_dict, edge_index_dict, domain_idx_to_attr, domain_sample_ids_full, samples_in_x_to_attr):
            # x_to_attr: torch.Tensor, the input tensor for the specific domain being attributed (e.g., mRNA_query.values)
            # static_x_dict: dict, contains all x_dict tensors
            # edge_index_dict: dict, contains all edge_index_dict tensors
            # domain_idx_to_attr: int, the index of this domain (e.g., 0 for mRNA)
            # domain_sample_ids_full: list or dict, self.dict_original_order (all samples common across all initial datasets)
            # samples_in_x_to_attr: list, the sample IDs specifically in x_to_attr (e.g., mRNA_query.index)

            x_dict_for_model = static_x_dict.copy()
            x_dict_for_model[domain_idx_to_attr] = x_to_attr # Replace the domain to be attributed with the actual x_to_attr from Captum

            _, _, model_output, model_output_sample_ids = model(x_dict_for_model, edge_index_dict, domain_sample_ids_full)

            # model_output_sample_ids is a list of sample IDs corresponding to the rows of model_output
            # Filter model_output to only include samples corresponding to samples_in_x_to_attr

            # Create a mapping from sample ID to its index in model_output_sample_ids for efficient lookup
            output_sample_id_to_idx = {sample_id: idx for idx, sample_id in enumerate(model_output_sample_ids)}

            # Get the indices in model_output that correspond to samples_in_x_to_attr
            indices_to_keep = []
            for sample_id in samples_in_x_to_attr:
                if sample_id in output_sample_id_to_idx:
                    indices_to_keep.append(output_sample_id_to_idx[sample_id])

            if not indices_to_keep:
                raise ValueError(f"No samples from the attributed domain {domain_idx_to_attr} ({len(samples_in_x_to_attr)} samples) found in the model's output ({len(model_output_sample_ids)} samples).")

            # Use advanced indexing to filter the tensor
            filtered_output = model_output[indices_to_keep]

            # Captum's IntegratedGradients expects a scalar output for each input instance for completeness check
            return filtered_output.sum(dim=1)


        # Prepare the data dictionaries for node features and edge indices.
        x_dict = {} # Stores tensors (modal_dg.x)
        edge_index_dict = {}
        original_sample_ids_per_domain = {} # Stores list of sample IDs for each new_dataset
        feature_names_per_domain = {} # Stores feature names for each new_dataset
        for i, modal_df in enumerate(new_datasets):
            model_name = modalities_names[i]
            modal_index = self.modalities_name_list.index(model_name)

            dataset = GraphDataset(
                self.neighbor_size,
                modal_df.values, # modal_df is a pandas DataFrame
                self.fused_networks[modal_index].values,
                transform=T.ToDevice(device),
            )
            modal_dg = dataset[0]

            x_dict[modal_index] = modal_dg.x
            edge_index_dict[modal_index] = modal_dg.edge_index
            original_sample_ids_per_domain[modal_index] = list(modal_df.index) # Store sample IDs
            feature_names_per_domain[modal_index] = list(modal_df.columns) # Store feature names

        # Compute feature importances using IntegratedGradients.
        feat_importances = {}
        df_list = [] # Initialize df_list here

        for domain_idx_to_attr in x_dict:
            x_input = x_dict[domain_idx_to_attr] # This is a tensor
            current_domain_original_samples = original_sample_ids_per_domain[domain_idx_to_attr]

            static_x = {k: x_dict[k] for k in x_dict} # Contains all x_dict tensors

            ig = IntegratedGradients(custom_forward)

            attributions, delta = ig.attribute(
                inputs=x_input,
                additional_forward_args=(static_x, edge_index_dict, domain_idx_to_attr, self.dict_original_order, current_domain_original_samples),
                return_convergence_delta=True
            )

            if domain_idx_to_attr not in feat_importances:
                feat_importances[domain_idx_to_attr] = []
            feat_importances[domain_idx_to_attr].append(attributions.detach().cpu().numpy())

        for domain_idx in feat_importances:

            # Concatenate along the first axis (nodes).
            feat_importances[domain_idx] = np.concatenate(feat_importances[domain_idx], axis=0)
            # Use original feature names for the columns
            df = pd.DataFrame(feat_importances[domain_idx], columns=feature_names_per_domain[domain_idx])
            df_list.append(df)

            # save the feature importance
            mod_name = self.modalities_name_list[domain_idx]
            csv_path = os.path.join(result_dir, f'{mod_name}_feat_importance.csv')
            df.to_csv(csv_path, index=False)
            print(df.shape)

            print(f"Saved feature importances for domain {mod_name} to {csv_path}")

        return df_list