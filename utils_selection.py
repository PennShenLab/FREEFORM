import random
import os
from collections import Counter
import pickle
import copy
from importlib import reload
import pandas as pd
from datetime import datetime
import re
import utils
reload(utils)
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV, LassoLarsCV, Lasso

_LLM_SELECTION_METHODS = ["llm_select", "iterative", "pyramid", "pyramid_two","iterative_self_consistency"]

def extract_snps_hearing_loss(snp):
    # More generic pattern to match SNPs
    pattern = r'\b[A-Za-z0-9][^\s]*[A-Za-z0-9]\b'
    
    match = re.search(pattern, snp)
    if match:
        return match.group()
    
    return None


def split_list(input_list, bucket_size):
    """Splits the input_list into buckets of roughly equal size."""
    if len(input_list) < bucket_size:
        return [input_list]
    n = len(input_list)
    num_buckets = round(n / bucket_size)
    adjusted_bucket_size = n // num_buckets
    remainder = n % num_buckets
    
    buckets = []
    start = 0
    for i in range(num_buckets):
        end = start + adjusted_bucket_size + (1 if i < remainder else 0)
        buckets.append(input_list[start:end])
        start = end
    
    return buckets

### For Heirarchical Selection
def generate_prompts(data_name, snps, bucket_size, top_n, final_iter = 0, model = "o1-preview"):
    """Generates prompts for each bucket of SNPs."""
    buckets = split_list(snps, bucket_size)
    prompts = []
    for bucket in buckets:
        random.shuffle(bucket) 
        features = ", ".join(bucket)
        if 'ancestry' in data_name:
            if final_iter == 0:
                prompt = f"You are a genetics expert with deep knowledge of genomic ancestry. From the following list, select the {top_n} most relevant SNPs for accurately predicting genomic ancestry (European, East Asian, African, American, or South Asian): {features}. List them one line after another."
                # prompt = f"You are a genetics expert. With all your knowledge, select the {top_n} most relevant SNPs for predicting genomic ancestry (European, East Asian, African, American, or South Asian). Many of these features are not classic AIMs but maybe be particularly useful: {features}\n\nList them one line after another."
            elif model == "o1-preview":
                prompt = f"You are a genetics expert with deep knowledge of genomic ancestry. From the following list, select the {top_n} most relevant SNPs for accurately predicting genomic ancestry (European, East Asian, African, American, or South Asian): {features}."
            else: 
                # prompt = f"You are a genetics expert. With all your knowledge, select the {top_n} most relevant SNPs for predicting genomic ancestry (European, East Asian, African, American, or South Asian): {features}.\n\nSolve the problem step by step."
                prompt = f"You are a genetics expert with deep knowledge of genomic ancestry. From the following list, select the {top_n} most relevant SNPs for accurately predicting genomic ancestry (European, East Asian, African, American, or South Asian): {features}.\n\nLet's solve the problem step by step."
        elif 'hearing_loss' in data_name:    
            #prompt = f"""Features: {features}\n\nTask Description: Does the subject have hereditary hearing loss?\n\nInstructions: You're an expert in genomics and hereditary hearing loss. From the list of SNPs, I want to select the most significant SNPs that will accurately predict hereditary hearing loss. Do not write code to this. Use your knowledge and reasoning to select the right features and make sure to only analyze the features from the provided list."""
            if model == "o1-preview":
                prompt = f"You are a genetics expert with deep knowledge of hereditary hearing loss. From the following list, select the {top_n} most significant SNPs for accurately predicting whether a subject has hereditary hearing loss or not: {features}"
            else:
                prompt = f"You are a genetics expert with deep knowledge of hereditary hearing loss. From the following list, select the {top_n} most significant SNPs for accurately predicting whether a subject has hereditary hearing loss or not: {features}\n\nSolve the problem step by step."
        else: 
            raise Exception("invalid data name")
        prompts.append(prompt)
    return prompts


def query_snps(data_name, prompts, df_columns, temperature, model, final_iter = 0, final_temperature = 0.7):
    """Queries GPT model with CoT multiple retries and combines results."""
    buckets = []
    results = utils.query_gpt(prompts, temperature=(final_temperature if final_iter == 1 else temperature), model=model)
    # If final_iter is 0 for genomic ancestry, there is no need to extract via LLM as chain-of-thought was not used
    if final_iter > 0 or "hearing_loss" in data_name:
        #print(results)
        results = utils.query_gpt([f"Extract the final list of 15 SNPs: {result}\nList them one line after another." for result in results], max_tokens=100)
    for result in results:
        #print("extracted results", result)
        result_snps = result.splitlines()
        if 'hearing_loss' in data_name:
            result_snps_extracted = [extract_snps_hearing_loss(snp) for snp in result_snps[:15] if extract_snps_hearing_loss(snp) and extract_snps_hearing_loss(snp) in df_columns]
        else:
            result_snps_extracted = [re.search(r'\brs\d+', snp).group() for snp in result_snps[:15] if re.search(r'\brs\d+', snp) and re.search(r'\brs\d+', snp).group() in df_columns]
        buckets.append(result_snps_extracted)
    return buckets


def heirarchical_selection(data_name, snps, df_columns, bucket_size=100, top_n=15, initial_retries=3, final_retries=10, temperature=0.5, final_temp = 0.7, model='gpt-4o-2024-05-13', final_model = "gpt-4o-2024-05-13"):
    """Main pipeline to identify the most relevant SNPs for predicting genomic ancestry."""

    while len(snps) > top_n:
        random.shuffle(snps) 
        final_snps = []
        num_of_buckets = max(round(len(snps)/bucket_size),1)
        bucket_results = [[] for _ in range(num_of_buckets)]
        print(f"Shuffling and partitioning {len(snps)} SNPs into {num_of_buckets} buckets...")

        for _ in range(initial_retries if num_of_buckets > 1 else final_retries):
            # Split snps into buckets 
            prompts = generate_prompts(data_name, snps, bucket_size, top_n, final_iter = num_of_buckets == 1)
            
            # Generates buckets of selected snps
            buckets = query_snps(data_name, prompts, df_columns, temperature, final_model if num_of_buckets == 1 else model, final_iter = num_of_buckets == 1, final_temperature= final_temp)
            #print(buckets)
            for i, bucket in enumerate(buckets): 
                bucket_results[i].extend(bucket)
        #print("---------okay iteration done. now let's gather the buckets and extract 15 from each.--------")
        # Self-consistency
        for bucket in bucket_results:
            #print(len(set(bucket)))
            #assert(len(bucket) > 15)
            snp_counter = Counter(bucket)
            top_15_snps = [snp for snp, _ in snp_counter.most_common(15)]
            #print(top_15_snps)
            final_snps.extend(top_15_snps)
        print(f"Reduced to: {final_snps}")
        snps = final_snps
        
    return snps

### End for Heirarchical Selection

def iterative_snp_selection_self_consistency(data_name, snps, model="gpt-4o-2024-05-13", top_k=15, num_consistency_checks=5):
    """
    Iteratively queries GPT to find the most significant SNP for predicting hereditary hearing loss or genomic ancestry,
    using self-consistency to select the most frequently identified SNP in each iteration.

    Parameters:
    snps : list
        A list of SNPs to query.
    model : str, optional (default="gpt-4o")
        The model to be used for querying GPT.
    top_k : int, optional (default=15)
        The number of top SNPs to select.
    num_consistency_checks : int, optional (default=5)
        The number of times to query GPT for consistency in each iteration.

    Returns:
    selected_snps : list
        A list of SNPs selected as the most significant in each iteration.
    """
    # Create a deep copy of the original SNP list to avoid modifying it
    snps_copy = copy.deepcopy(snps)
    selected_snps = []
    
    # Define the prompt based on the data_name
    if "hearing_loss" in data_name:
        prompt = "Out of the following list of SNPs, give me the single most significant SNP for predicting hereditary hearing loss:"
    else:
        prompt = "Out of the following list of SNPs, give me the single most significant SNP for predicting genomic ancestry (American, South Asian, African, East Asian, European):"
    
    while len(selected_snps) < top_k and snps_copy:
        consistency_results = []
        try:
            if len(selected_snps) < 3:
                runs = 1
            elif len(selected_snps) < 11: 
                runs = 3
            else: 
                runs = num_consistency_checks
            for _ in range(runs):                    
                # Query GPT to find the most significant SNP
                response = utils.query_gpt([f"{prompt} {snps_copy}"], model=model, temperature=0.7)
                #print(response[0])
                snp = utils.query_gpt([f"From the text, extract the SNP that is concluded at the end and give only the SNP name: {response[0]}"], model=model)
                most_significant_snp = snp[0].strip()  # Assuming the response is returned as a list with the first element being the SNP
                if most_significant_snp in snps_copy: 
                    consistency_results.append(most_significant_snp)
                #print(f"Consistency check result: {most_significant_snp}")
            
            # Identify the most frequently suggested SNP
            most_common_snp = Counter(consistency_results).most_common(1)[0][0]
            print(f"Most consistent SNP: {most_common_snp}")
            if most_common_snp not in snps_copy:
                raise Exception("SNP not from the list")
            # Add the identified SNP to the selected_snps list
            selected_snps.append(most_common_snp)

            # Remove the identified SNP from the copy list
            snps_copy.remove(most_common_snp)
        
        except Exception as e:
            print(e)
            while True:
                try:
                    if len(selected_snps) < 11: 
                        runs = 3
                    else: 
                        runs = num_consistency_checks
                    for _ in range(runs):
                        response = utils.query_gpt([f"{prompt} {snps_copy}. Do not select SNPs in the list {selected_snps} as they have already been selected and removed."], temperature=0.5, model=model)
                        #print(response[0])
                        snp = utils.query_gpt([f"From the text, extract the SNP that is concluded at the end and give only the SNP name: {response[0]}"], model=model)
                        most_significant_snp = snp[0].strip()
                        if most_significant_snp in snps_copy:
                            consistency_results.append(most_significant_snp)
                            #print(f"Consistency check result: {most_significant_snp}")
                    
                    # Identify the most frequently suggested SNP
                    most_common_snp = Counter(consistency_results).most_common(1)[0][0]
                    print(f"Most consistent SNP: {most_common_snp}")

                    if most_common_snp not in snps_copy:
                        raise Exception("SNP not from the list")
                    
                    # Add the identified SNP to the selected_snps list
                    selected_snps.append(most_common_snp)

                    # Remove the identified SNP from the copy list
                    snps_copy.remove(most_common_snp)
                    break
                except Exception as ex:
                    print(ex)
    
    return selected_snps

def iterative_snp_selection(data_name, snps, model="gpt-4o-2024-05-13", top_k = 15):
    """
    Iteratively queries GPT to find the most significant SNP for predicting hereditary hearing loss,
    removes it from a deep copy of the original list, and continues until the list is empty.

    Parameters:
    snps : list
        A list of SNPs to query.
    model : str, optional (default="gpt-4o")
        The model to be used for querying GPT.

    Returns:
    selected_snps : list
        A list of SNPs selected as the most significant in each iteration.
    """
    # Create a deep copy of the original SNP list to avoid modifying it
    snps_copy = copy.deepcopy(snps)
    selected_snps = []
    if "hearing_loss" in data_name:
        prompt = "Out of the following list of SNPs, give me the single most significant SNP for predicting hereditary hearing loss:"
    else: 
        prompt = "Out of the following list of SNPs, give me the single most significant SNP for predicting genomic ancestry (American, South Asian, African, East Asian, European):"
    while len(selected_snps) <= top_k:
        try: 
            # Query GPT to find the most significant SNP
            response = utils.query_gpt([f"{prompt} {snps_copy}"], model=model)
            print(response[0])
            snp = utils.query_gpt([f"From the chain of thought, extract the SNP that is concluded at the end and give only the SNP name: {response[0]}"])
            most_significant_snp = snp[0]  # Assuming the response is returned as a list with the first element being the SNP
            print(most_significant_snp)
            # Add the identified SNP to the selected_snps list
            selected_snps.append(most_significant_snp)

            # Remove the identified SNP from the copy list
            snps_copy.remove(most_significant_snp)
        except Exception as e: 
            while True:
                try: 
                    print(e)
                    response = utils.query_gpt([f"{prompt} {snps_copy}. Do not select SNPs in the list {selected_snps} as they have already been selected and removed from the list."], temperature=0.5,model=model)
                    print(response[0])
                    snp = utils.query_gpt([f"From the chain of thought, extract the SNP that is concluded at the end and give only the SNP name: {response[0]}"])
                    most_significant_snp = snp[0]  # Assuming the response is returned as a list with the first element being the SNP
                    print(most_significant_snp)
                    if most_significant_snp not in snps:
                        raise Exception("SNP not from the list")
                    # Add the identified SNP to the selected_snps list
                    selected_snps.append(most_significant_snp)

                    # Remove the identified SNP from the copy list
                    snps_copy.remove(most_significant_snp)
                    break
                except Exception as ex:
                    print(ex)

    return selected_snps

def iterative_tree_of_thought(data_name, feature_selection_pool, top_k=15):
    pass

def llm_select(data_name, snps,model='gpt-4o-2024-05-13', top_k = 15):
    prompt = {}
    if "ancestry" in data_name:
        prompt["system"] = f"""Given a list of features, rank them according to their importances in predicting whether an individual is genetically from East Asian, European, American, South Asian, or African ancestry. The ranking should be in descending order, starting with the most important feature."""
        prompt["user"] = f"Rank all ⟨number of concepts⟩ features in the following list: \"{snps}\". Your response should be a numbered list with each item on a new line. For example: 1. foo 2. bar 3. baz Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking."

    elif "hearing_loss" in data_name:
        prompt["system"] = f"""Given a list of features, rank them according to their importances in predicting if an individual has hereditary hearing loss. The ranking should be in descending order, starting with the most important feature."""
        prompt["user"] = f"Rank all ⟨number of concepts⟩ features in the following list: \"{snps}\". Your response should be a numbered list with each item on a new line. For example: 1. foo 2. bar 3. baz Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking."
    else: 
        raise Exception("invalid data_name")

    response = utils.query_full_gpt([prompt], model=model)
    print(response[0])
    snps = response[0].splitlines()
    cleaned_snps = []
    for snp in snps:
        print(snp)
        cleaned_snps.append(snp.split(" ")[1])
    return cleaned_snps[:top_k]
    
def lasso_feature_selection(X, y, top_k=15, cv = 4, data_name = "hearing_loss"):
    # Use LassoCV to perform cross-validation and select the best alpha
    alphas = [1e-4,1e-3,1e-2,1e-1,1,10]
    if 'ancestry' in data_name:
        # ancestry has too many columns
        lasso_cv = LassoLarsCV(cv=cv).fit(X,y)
    else: 
        lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=0).fit(X, y)
    # Fit Lasso with the best alpha
    best_alpha = lasso_cv.alpha_
    #print(f"Selected Alpha: {best_alpha}")
    
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X, y)
    
    # Get the coefficients and select the top k features
    lasso_coef = np.abs(lasso.coef_)
    top_k_indices = np.argsort(lasso_coef)[-top_k:][::-1]  # Sort by absolute value and get top k
    
    return X.columns[top_k_indices]

def ml_rank_importance_selection(X, y, model, criteria, top_k=15, seed=42, scoring='roc_auc_ovr'):
    print(f"Starting feature selection with {model} model using {criteria} criteria.")
    
    # Initialize the model
    if model == "rf":
        base_model = RandomForestClassifier(random_state=seed)
    elif model == "lr":
        base_model = LogisticRegression(random_state=seed)
    else:
        raise Exception("Invalid model name. Use 'rf' for Random Forest or 'lr' for Logistic Regression.")
    
    # Fit the base model
    base_model.fit(X, y)
    
    if criteria == 'gini' and model == 'rf':
        # Stage 1: Gini Importance (Random Forest specific)
        print("Calculating Gini importance for feature selection...")
        feature_importances = base_model.feature_importances_
        top_k_idx = np.argsort(feature_importances)[-top_k:]  # Get indices of top_k features
        selected_features = top_k_idx
        print(f"Reduced feature set to {top_k} features based on Gini importance.")

    elif criteria == 'permutation':
        # Stage 2: Permutation Importance
        print("Calculating permutation importance for feature selection...")
        perm_importance = permutation_importance(base_model, X, y, n_repeats=10, random_state=seed, scoring=scoring)
        top_k_idx = np.argsort(perm_importance.importances_mean)[-top_k:]  # Get indices of top_k features
        selected_features = top_k_idx
        print(f"Reduced feature set to {top_k} features based on permutation importance.")
    
    else:
        raise Exception("Invalid criteria. Use 'gini' for Gini Importance or 'permutation' for Permutation Importance.")
    
    return X.columns[selected_features]
    
    
def pca_feature_selection(X, y=None, top_k=15, cv=None):

    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Perform PCA
    pca = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]))  # n_components = min(n_samples, n_features)
    pca.fit(X_scaled)

    # Step 3: Get the PCA loadings (components_)
    loadings = pca.components_.T  # Transpose to get features by components

    # Step 4: Calculate the importance of each feature for the first principal component
    importance = np.abs(loadings[:, 0])

    # Step 5: Get the indices of the top k features
    top_k_indices = np.argsort(importance)[-top_k:][::-1]  # Sort and take top k, in descending order

    # Step 6: Get the names of the top k features
    top_k_features = X.columns[top_k_indices]

    return top_k_features.tolist()

def pca_weighted_sum_feature_selection(X, y, top_k=15, cv=None, num_components=None):
    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Determine the number of components for PCA
    if num_components is None:
        num_components = min(X_scaled.shape[0], X_scaled.shape[1])

    # Step 3: Perform PCA
    pca = PCA(n_components=num_components)
    pca.fit(X_scaled)

    # Step 4: Get the PCA loadings
    loadings = pca.components_.T  # Transpose to get features by components
    selected_loadings = loadings[:, :num_components]

    # Step 5: Perform the weighted sum of loadings
    explained_variance = pca.explained_variance_ratio_
    weighted_sum_loadings = np.sum(selected_loadings * explained_variance[:num_components], axis=1)

    # Step 6: Get the indices of the top k features
    top_k_indices = np.argsort(weighted_sum_loadings)[-top_k:][::-1]

    # Step 7: Get the names of the top k features
    top_k_features = X.columns[top_k_indices]

    return top_k_features.tolist()  

def pca_m_contribution_selection(X, y, top_k=15, num_components=None):
    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Determine the number of components for PCA
    if num_components is not None:
        num_components = min(X_scaled.shape[0], X_scaled.shape[1])
        print(num_components)


    # Step 3: Perform PCA
    pca = PCA(n_components=num_components)
    pca.fit(X_scaled)

    # Step 4: Get the PCA eigenvectors (loadings)
    # The components_ attribute is already the eigenvectors of the covariance matrix
    loadings = pca.components_.T  # Transpose to get features by components
    selected_loadings = loadings[:, :num_components]  # Select the top m eigenvectors

    # Step 5: Calculate the contribution of each feature
    # c_j = sum_{p=1}^m |V_pj|
    feature_contributions = np.sum(np.abs(selected_loadings), axis=1)

    # Step 6: Sort c_j in descending order and get the indices of the top k features
    top_k_indices = np.argsort(feature_contributions)[-top_k:][::-1]

    # Step 7: Get the names of the top k features
    if isinstance(X, pd.DataFrame):
        top_k_features = X.columns[top_k_indices]
    else:
        top_k_features = top_k_indices  # Return indices if X has no column names

    return top_k_features.tolist()

def forward_feature_selection_sklearn(X, y, top_k=15, cv=5, seed=0, scoring='roc_auc_ovr', model = "lr", direction="forward"):
    """
    Perform forward feature selection using sklearn's SequentialFeatureSelector with Random Forest.
    
    Parameters:
    - X: DataFrame, feature set.
    - y: Series or array, target variable.
    - top_k: int, number of top features to select.
    - cv: int, number of cross-validation folds.
    - random_state: int, random state for reproducibility.
    - scoring: str, scoring metric to optimize during feature selection.
    
    Returns:
    - selected_features: list, the features selected by forward selection.
    - final_model: RandomForestClassifier, the trained model using the selected features.
    - roc_auc: float, the ROC AUC score on the test set using the selected features.
    - accuracy: float, the accuracy score on the test set using the selected features.
    """
    # Initialize the RandomForest classifier
    print(model)
    if model == "rf":
        model = RandomForestClassifier(seed=seed)
    elif model == "lr":
        model = LogisticRegression(seed=seed)
    else: 
        raise Exception("invalid model name")
    
    sfs = SequentialFeatureSelector(model, n_features_to_select=top_k, direction=direction, cv=cv, scoring=scoring)
    print("Fitting....")
    # Fit the SFS
    sfs.fit(X, y)

    # Get the selected features
    selected_features = X.columns[sfs.get_support()]

    return selected_features


def data_driven_feature_selection_pipeline(X, y, initial_reduction = 1000, second_reduction = 50, top_k=15, cv=5, seed=0, scoring='roc_auc_ovr', model="lr", direction="forward"):
    """
    Perform feature selection using a multi-stage approach: Chi-square test, permutation-based feature importance, 
    and forward selection using sklearn's SequentialFeatureSelector.
    
    Parameters:
    - X: DataFrame, feature set (with 10,000 SNPs).
    - y: Series or array, target variable.
    - top_k: int, number of top features to select (default: 15).
    - cv: int, number of cross-validation folds (default: 5).
    - seed: int, random seed for reproducibility.
    - scoring: str, scoring metric to optimize during feature selection (default: 'roc_auc_ovr').
    - model: str, model to use for feature selection ('lr' for Logistic Regression, 'rf' for Random Forest).
    - direction: str, direction of Sequential Feature Selection ('forward' or 'backward').
    
    Returns:
    - selected_features: list, the features selected by forward selection.
    """
    
    # Stage 1: Chi-Square Test (filtering down to top 100 features)
    print("Performing Chi-Square test for feature selection...")
    
    chi2_selector = SelectKBest(chi2, k=initial_reduction)  # Reducing from 10,000 SNPs to 1000
    X_reduced = chi2_selector.fit_transform(X, y)
    selected_initial_features = X.columns[chi2_selector.get_support()]
    print(f"Reduced feature set to {X_reduced.shape[1]} features.")
    
    # Stage 2: Ranking by Importance (reducing to top 50 features)
    print("Calculating gini importance for further reduction...")
    
    if model == "rf":
        base_model = RandomForestClassifier(random_state=seed)
    elif model == "lr":
        base_model = LogisticRegression(random_state=seed, max_iter=500)
    else:
        raise Exception("Invalid model name")
    
    base_model.fit(X_reduced, y)
    
    # Calculate gini importance
    feature_importances = base_model.feature_importances_
    top_M_idx = np.argsort(feature_importances)[-second_reduction:]  # Get indices of top_k features
    
    # Calculate permutation importance
    # perm_importance = permutation_importance(base_model, X_reduced, y, n_repeats=10, random_state=seed, scoring=scoring)
    # Get the indices of the top M features
    # top_M_idx = np.argsort(perm_importance.importances_mean)[-second_reduction:]
    X_top_M = X_reduced[:, top_M_idx]
    selected_M_features = selected_initial_features[top_M_idx]
    print(f"Reduced feature set to {X_top_M.shape[1]} features based on permutation importance.")
    
    # Stage 3: Sequential Feature Selection (reducing to top 15 features)
    print("Performing Sequential Feature Selection...")
    
    sfs = SequentialFeatureSelector(base_model, n_features_to_select=top_k, direction=direction, cv=cv, scoring=scoring)
    
    sfs.fit(X_top_M, y)
    
    # Get the selected features from the final model
    selected_features = selected_M_features[sfs.get_support()]
    
    print(f"Final selected features: {selected_features}")
    
    return selected_features


def evaluate_feature_selection(data_name, selection_method, seeds, shots, top_k=15, manually_selected_features=None, feature_selection_pool=None, return_stored_results = False, old_results = False, use_grid_search = True, load_specific_pkl = None, model = None):
    """
    Parameters:
    ----------
    data_name : str
        The name of the dataset being used for feature selection.
    
    selection_method : str
        The method used for feature selection. Possible values include:
        - "llm_select": LLM-based selection.
        - "iterative": Iterative SNP selection.
        - "pyramid": Hierarchical selection with iteration.
        - "iterative_self_consistency": Iterative selection with consistency checks.
        - "lasso", "forward_selection", "backward_selection", "pca", "manual": Classical feature selection methods.
    
    seeds : list[int]
        A list of random seeds to use for reproducibility in feature selection and model training.
    
    shots : list[int]
        A list of integers representing the number of training samples used in few-shot learning experiments.
    
    top_k : int, optional, default=15
        The number of top features to select from each selection method.
    
    manually_selected_features : list, optional, default=None
        Predefined list of features to be used if the "manual" selection method is chosen.
    
    feature_selection_pool : list, optional, default=None
        The pool of features (variants) from which relevant features are selected by the selection methods.
    
    return_stored_results : bool, optional, default=False
        If True, attempts to load all previously saved results (e.g., Logistic Regression and Random Forest results) from a file, instead of recalculating them.
    
    old_results : bool, optional, default=False
        Flag to indicate if old results are to be used in some way; the exact usage can be determined based on how this flag interacts with loading or recalculating results.
    
    Returns:
    -------
    all_log_reg_results : dict
        Dictionary containing the evaluation results (accuracy, AUC, F1 score) for Logistic Regression.
    
    all_rf_results : dict
        Dictionary containing the evaluation results (accuracy, AUC, F1 score) for Random Forest.
    """
    if model is None:
        output_file = f"../data/selected_features/feature_selection_{data_name}_{selection_method}_{shots}_{seeds}_{top_k}.pkl"
    else: 
        output_file = f"../data/selected_features/feature_selection_{data_name}_{selection_method}_{shots}_{seeds}_{top_k}_{model}.pkl"
    if old_results:
        output_file = f"../data/selected_features/feature_selection_{data_name}_{selection_method}.pkl"
    elif load_specific_pkl is not None:
        output_file = output_file.split('.pkl')[0] + f"{load_specific_pkl}.pkl"
    # Initialize containers for results
    all_log_reg_results = {}
    all_rf_results = {}
    all_feature_selections = {}
    loaded_features = False
    subset_found = False
    existing_seeds = []
    file_name = f"../data/{data_name}.csv"
    preloaded_df = pd.read_csv(file_name)
    print(output_file)
    
    if os.path.isfile(output_file):
        print(f"Loading full results...")
        saved_log_reg_results, saved_rf_results, all_feature_selections = load_saved_results(output_file, selection_method)
        existing_seeds = seeds
        loaded_features = True
        subset_found = True
        if return_stored_results:
            return saved_log_reg_results, saved_rf_results, all_feature_selections
        # In the case that we're not returning stored results although it exists, we're either (1) just repeating ML analysis OR (2) repeating ML analysis on new manual features
        elif selection_method == "manual": 
            loaded_features = all_feature_selections[4][0] == manually_selected_features
            existing_seeds = []
            subset_found = False
    else:
        # Try to find an existing result file with a subset of seeds
        for i in range(len(seeds), 0, -1):  # Start with the full list and reduce
            seeds_subset = seeds[:i]
            if model is None:
                subset_output_file = f"../data/selected_features/feature_selection_{data_name}_{selection_method}_{shots}_{seeds_subset}_{top_k}.pkl"
            else:
                subset_output_file = f"../data/selected_features/feature_selection_{data_name}_{selection_method}_{shots}_{seeds_subset}_{top_k}_{model}.pkl"
                
            if os.path.isfile(subset_output_file) or os.path.isfile(output_file):
                print(f"Found existing file with seeds: {seeds_subset}. Loading partial results...")
                saved_log_reg_results, saved_rf_results, all_feature_selections = load_saved_results(subset_output_file, selection_method)
                existing_seeds = seeds_subset
                if i == len(seeds):
                    loaded_features = True
                    if return_stored_results:
                        return saved_log_reg_results, saved_rf_results, all_feature_selections
                subset_found = True
                break
    
    # Determine the new seeds to process
    new_seeds = [seed for seed in seeds if seed not in existing_seeds]
    print(f"New seeds to process: {new_seeds}")    

    # Handle LLM-selection separately since it doesn't rely on shots
    if selection_method in _LLM_SELECTION_METHODS:
        if subset_found:
            for seed in new_seeds:
                all_feature_selections[seed] = perform_feature_selection(selection_method, data_name, feature_selection_pool, top_k, manually_selected_features, model=model)
        else:
            all_feature_selections = {seed: perform_feature_selection(selection_method, data_name, feature_selection_pool, top_k, manually_selected_features, model=model) for seed in seeds}

    # If applies, do ML-based feature selection
    for shot in shots:
        if len(new_seeds) > 0:
            print(f"------------- Doing Feature selection for {shot}-shots -------------")
            feature_selections_for_shot = {}

            for seed in new_seeds:
                print(f"Seed: {seed}")
                # Split data and perform feature selection if needed
                df, X_train, X_test, y_train, y_test, target_attribute, label_list, _ = utils.get_dataset(data_name, shot, seed, preloaded_df=preloaded_df)
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)

                # Feature selection based on the method
                if selection_method in _LLM_SELECTION_METHODS:
                    selected_features = all_feature_selections[seed]
                elif loaded_features: 
                    selected_features = all_feature_selections[shot][seed]
                else: 
                    print(len(X_train))
                    selected_features = perform_feature_selection(selection_method, data_name, feature_selection_pool, top_k, manually_selected_features, X_train=X_train, y_train=y_train, shot=shot, seed=seed, model=model)
                
                # Store selected features
                feature_selections_for_shot[seed] = selected_features
            
            if selection_method not in _LLM_SELECTION_METHODS:
                all_feature_selections[shot] = feature_selections_for_shot

    # Downstream Evaluation of selected features
    for shot in shots:
        print(f"------------- Doing ML Analysis for {shot}-shots -------------")
        log_reg_results = {'accuracy': [], 'auc': [], 'f1': []}
        rf_results = {'accuracy': [], 'auc': [], 'f1': []}

        
        # If the user requested the loading of previous results, but we only have a subset, we should only calculate the new seeds again
        temp_seeds = new_seeds
        if not return_stored_results:
            print("Re-doing complete ML downstream analysis...")
            temp_seeds = seeds
        else: 
            print(f"redoing downstream analysis for seeds {temp_seeds}...")
        if len(temp_seeds) > 0:
            for seed in temp_seeds:
                print(f"Seed: {seed}")
                # Split data and perform feature selection if needed
                df, X_train, X_test, y_train, y_test, target_attribute, label_list, _ = utils.get_dataset(data_name, shot, seed,preloaded_df=preloaded_df)
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)

                # Recall Selected Features
                if selection_method in _LLM_SELECTION_METHODS:
                    selected_features = all_feature_selections[seed]
                else: 
                    selected_features = all_feature_selections[shot][seed]

                X_train = X_train[selected_features]
                X_test_selected = X_test[selected_features]
                
                # Evaluate models for both Logistic Regression and Random Forest
                accuracy_log_reg, auc_log_reg, f1_log_reg = evaluate_model(X_train, X_test_selected, y_train, y_test, seed, model_type='log_reg', use_grid_search=use_grid_search)
                accuracy_rf, auc_rf, f1_rf = evaluate_model(X_train, X_test_selected, y_train, y_test, seed, model_type='rf',use_grid_search=use_grid_search)

                # Store the ML results
                log_reg_results['accuracy'].append(accuracy_log_reg)
                log_reg_results['auc'].append(auc_log_reg)
                log_reg_results['f1'].append(f1_log_reg)

                rf_results['accuracy'].append(accuracy_rf)
                rf_results['auc'].append(auc_rf)
                rf_results['f1'].append(f1_rf)

            # Store the results for this shot
            all_log_reg_results[shot] = log_reg_results
            all_rf_results[shot] = rf_results
        
    # Save the final feature selections and results to a file
    params = {'seeds': seeds, 'shots': shots, 'top_k': top_k, 'data_name': data_name, 'selection_method': selection_method}
    
    if os.path.isfile(output_file):
        output_file = f"../data/selected_features/feature_selection_{data_name}_{selection_method}_{shots}_{seeds}_{top_k}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pkl"
    save_results(output_file, all_feature_selections, all_log_reg_results, all_rf_results, params)
    
    return all_log_reg_results, all_rf_results, all_feature_selections

## For Evaluating Feature Selection
def save_results(output_file, all_feature_selections, log_reg_results, rf_results, params):
    """Save feature selection and ML results to a file."""
    with open(output_file, 'wb') as f:
        pickle.dump({
            'feature_selections': all_feature_selections,
            'params': params,
            'log_reg_results': log_reg_results,
            'rf_results': rf_results
        }, f)

def load_saved_results(output_file, selection_method):
    """Load saved feature selection and ML results if available."""
    print("Results and feature selections have already been calculated and saved.... loading selected features")
    with open(output_file, 'rb') as f:
        saved_data = pickle.load(f)
        all_feature_selections = saved_data['feature_selections']
        if selection_method in ["llm_select", "iterative", "pyramid", "iterative_self_consistency"]:
            all_feature_selections[4] = saved_data['feature_selections'][0]
    return saved_data['log_reg_results'], saved_data['rf_results'], all_feature_selections

def perform_feature_selection(selection_method, data_name, feature_selection_pool, top_k, manually_selected_features, X_train = None, y_train = None, shot = None, seed = None, model=None):
    """Perform feature selection based on the specified method."""
    print(f"Performing feature selection... {selection_method} on {data_name}")
    if selection_method == "pyramid":
        if "hearing_loss" in data_name:
            return heirarchical_selection(data_name, feature_selection_pool, feature_selection_pool, top_n=top_k, temperature=0.3, initial_retries=3, final_retries=20, bucket_size=50, model=model)
        else:
            return heirarchical_selection(data_name, feature_selection_pool, feature_selection_pool, top_n=top_k, temperature=0.3, initial_retries=3, final_retries=20, bucket_size=75, model=model)
    if selection_method == "pyramid_two":
        if "hearing_loss" in data_name:
            return heirarchical_selection(data_name, feature_selection_pool, feature_selection_pool, top_n=top_k, temperature=0.3, final_temp=1, initial_retries=3, final_retries=3, bucket_size=50, model=model, final_model="o1-preview")
        else:
            return heirarchical_selection(data_name, feature_selection_pool, feature_selection_pool, top_n=top_k, temperature=0.3, final_temp=1, initial_retries=3, final_retries=3, bucket_size=75, model=model, final_model="o1-preview")
    elif selection_method == "llm":
        return iterative_snp_selection(data_name, feature_selection_pool, top_k=top_k, model=model)
    elif selection_method == "iterative":
        return iterative_snp_selection(data_name, feature_selection_pool, top_k=top_k, model=model)
    elif selection_method == "iterative_self_consistency":
        return iterative_snp_selection_self_consistency(data_name, feature_selection_pool, top_k=top_k, num_consistency_checks=4, model=model)
    elif selection_method == "iterative_tree_of_thought":
        return iterative_tree_of_thought(data_name, feature_selection_pool, top_k=top_k)
    elif selection_method == "llm_select":
        return llm_select(data_name, feature_selection_pool, top_k=top_k, model=model)
    elif selection_method == "lasso":
        return lasso_feature_selection(X_train, y_train, top_k=top_k, data_name=data_name)
    elif selection_method == "forward_selection":
        return forward_feature_selection_sklearn(X_train, y_train, top_k=15,cv = 2 if shot <= 10 else 4, seed=seed,direction="forward", model=model)
    elif selection_method == "backward_selection":
        return forward_feature_selection_sklearn(X_train, y_train, top_k=15, cv = 2 if shot <= 10 else 4,seed=seed, direction="backward", model=model)
    elif selection_method == "pca":
        return pca_feature_selection(X_train, y_train, top_k=top_k)
    elif selection_method == "pca_weighted_sum": 
        return pca_weighted_sum_feature_selection(X_train, y_train, top_k=top_k)
    elif selection_method == "pca_m":
        return pca_m_contribution_selection(X_train, y_train, top_k=15, num_components=4)
    elif selection_method == "gini":
        return ml_rank_importance_selection(X_train, y_train, top_k=top_k, model=model, criteria="gini", seed=seed)
    elif selection_method == "permutation":
        return ml_rank_importance_selection(X_train, y_train, top_k=top_k, model=model, criteria="permutation", seed=seed)
    elif selection_method == "data_pipeline_forward":
        if "hearing_loss" in data_name:
            return data_driven_feature_selection_pipeline(X_train, y_train, initial_reduction = 100, second_reduction = 25, top_k=top_k, cv = 2 if shot <= 10 else 4, seed =seed, scoring='roc_auc_ovr', model=model, direction="forward")
        elif "ancestry" in data_name:
            return data_driven_feature_selection_pipeline(X_train, y_train, initial_reduction = 1000, second_reduction = 25, top_k=top_k, cv = 2 if shot <= 10 else 4, seed =seed, scoring='roc_auc_ovr', model=model, direction="forward")
    elif selection_method == "data_pipeline_backward":
        if "hearing_loss" in data_name:
            return data_driven_feature_selection_pipeline(X_train, y_train, initial_reduction = 100, second_reduction = 25, top_k=top_k, cv = 2 if shot <= 10 else 4, seed =seed, scoring='roc_auc_ovr', model=model, direction="backward")
        elif "ancestry" in data_name:
            return data_driven_feature_selection_pipeline(X_train, y_train, initial_reduction = 1000, second_reduction = 25, top_k=top_k, cv = 2 if shot <= 10 else 4, seed =seed, scoring='roc_auc_ovr', model=model, direction="backward")
    elif selection_method == "manual": 
        return manually_selected_features
    else:
        raise Exception("Invalid selection method")

def evaluate_model(X_train, X_test_selected, y_train, y_test, seed, model_type='log_reg', use_grid_search=True):
    """Train and evaluate a machine learning model (Logistic Regression or Random Forest)."""
    print(f"Evaluating {model_type} for seed {seed}...")
    if model_type == 'log_reg':
        params = {'C': [0.1, 1, 10], 'solver': ['saga', 'lbfgs']}
        model = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        model = RandomForestClassifier(random_state=seed)

    if use_grid_search:
        grid_search = GridSearchCV(model, params, cv=2 if len(X_train) <= 10 else 4, scoring='roc_auc_ovr')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test_selected)
    pred_probs = best_model.predict_proba(X_test_selected)

    # Calculate AUC based on binary or multiclass classification
    if len(np.unique(y_train)) == 2:
        pred_probs = pred_probs[:, 1]
        auc = roc_auc_score(y_test, pred_probs)
    else:
        auc = roc_auc_score(y_test, pred_probs, multi_class='ovr')

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, auc, f1


# This is for feature selection
def print_results(method_name, data_name, log_reg_results, rf_results, shots):
    print(f"Results for k-shot {method_name} feature selection on k-shot classification for: {data_name}\n")
    results = []

    for shot in shots:
        avg_log_reg_accuracy = np.mean(log_reg_results[shot]['accuracy'])
        avg_log_reg_auc = np.mean(log_reg_results[shot]['auc'])
        avg_log_reg_f1 = np.mean(log_reg_results[shot]['f1'])
        std_log_reg_accuracy = np.std(log_reg_results[shot]['accuracy'])
        std_log_reg_auc = np.std(log_reg_results[shot]['auc'])
        std_log_reg_f1 = np.std(log_reg_results[shot]['f1']) 
        
        avg_rf_accuracy = np.mean(rf_results[shot]['accuracy'])
        avg_rf_auc = np.mean(rf_results[shot]['auc'])
        avg_rf_f1 = np.mean(rf_results[shot]['f1'])
        std_rf_accuracy = np.std(rf_results[shot]['accuracy']) 
        std_rf_auc = np.std(rf_results[shot]['auc'])
        std_rf_f1 = np.std(rf_results[shot]['f1']) 

        results.append([
            shot, 
            "Logistic Regression", 
            f"{avg_log_reg_accuracy:.4f}", f"{std_log_reg_accuracy:.2f}", 
            f"{avg_log_reg_auc:.4f}", f"{std_log_reg_auc:.2f}", 
            f"{avg_log_reg_f1:.4f}", f"{std_log_reg_f1:.2f}"
        ])
        results.append([
            shot, 
            "Random Forest", 
            f"{avg_rf_accuracy:.4f}", f"{std_rf_accuracy:.2f}", 
            f"{avg_rf_auc:.4f}", f"{std_rf_auc:.2f}", 
            f"{avg_rf_f1:.4f}", f"{std_rf_f1:.2f}"
        ])
    
    headers = ["SHOT", "Model", "Avg Accuracy", "Std", "Avg AUC", "Std", "Avg F1 Score", "Std"]
    print(tabulate(results, headers=headers, tablefmt="pretty"))
    print("\n")


def plot_results_multiple(data_name, ml_name, results_dict_list, shots_list, titles_list=None, save_path='./plots', plot_type='feature_selection', save=False):
    # Number of plots
    n_plots = len(results_dict_list)
    
    # Determine grid size (e.g., for 4 plots, use 2x2 grid)
    if n_plots <= 3:
        nrows = 1
        ncols = n_plots
    else:
        nrows = 2
        ncols = int(np.ceil(n_plots / nrows))

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*4), squeeze=True)
    
    # Update font sizes
    plt.rcParams.update({
        'font.size': 14,            # Default text size
        'axes.titlesize': 16,       # Title font size
        'axes.labelsize': 14,       # X and Y label size
        'xtick.labelsize': 12,      # X-tick label size
        'ytick.labelsize': 12,      # Y-tick label size
        'legend.fontsize': 12       # Legend font size
    })
    
    # Collect legend handles and labels
    legend_entries = []

    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()
    
    # Iterate over each plot, using the corresponding shots list
    for idx, (results_dict, shots, ax) in enumerate(zip(results_dict_list, shots_list, axes_flat)):
        # Prepare x positions for equal spacing
        x_positions = range(len(shots))
        
        for method, results in results_dict.items():
            avg_accuracies = [np.mean(results[shot]['auc']) for shot in shots]
            wrapped_method = method.replace('+ ', '+\n')
            
            # Determine line style based on method name
            if 'FREEFORM' in method:
                linestyle = '-'
            else:
                linestyle = ':'  # Dotted line
            
            # Plot using x_positions for equal spacing
            line, = ax.plot(x_positions, avg_accuracies, marker='o', linestyle=linestyle, label=wrapped_method)
            
            # Collect legend entries as a tuple (wrapped_method, line)
            legend_entries.append((wrapped_method, line))
        
        # Set x-ticks to x_positions with labels shots
        ax.set_xticks(x_positions)
        ax.set_xticklabels(shots)
        
        # Set titles if provided
        if titles_list and idx < len(titles_list):
            ax.set_title(titles_list[idx])
        else:
            ax.set_title(f'Plot {idx+1}')
 
        ax.set_xlabel('Number of Shots')
        # Set y-axis label only for leftmost subplots
        if not (ncols > 1 and (idx % ncols) != 0):
            ax.set_ylabel('AUROC')
        ax.grid(True)
    
    # Hide any unused subplots
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)
    
    # Remove duplicate legend entries
    unique_legend_entries = {}
    for label, handle in legend_entries:
        if label not in unique_legend_entries:
            unique_legend_entries[label] = handle

    # Separate legend entries into two groups
    non_freeform_entries = []
    freeform_entries = []
    for label, handle in unique_legend_entries.items():
        if 'FREEFORM' in label:
            freeform_entries.append((label, handle))
        else:
            non_freeform_entries.append((label, handle))

    # Combine entries with non-FREEFORM first
    ordered_legend_entries = non_freeform_entries + freeform_entries

    # Unpack labels and handles
    legend_labels, legend_handles = zip(*ordered_legend_entries)

    # Adjust layout to make room for the legend at the bottom
    fig.tight_layout(rect=[0, 0.12, 1, 1])  # Leave more space at the bottom

    # Calculate number of columns for legend to make it two rows
    ncol = int(np.ceil(len(legend_labels) / 2))
    
    # Add a single legend at the bottom
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 0.02))
    
    # Save the plot as a PDF
    file_name = f'{data_name}_{ml_name}_{plot_type}.pdf'
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, file_name), format='pdf', bbox_inches='tight')
    
    plt.show()
    
    
def plot_results(data_name, ml_name, results_dict, shots, save_path='./plots', type='feature_selection', save=False):
    
    
    # Use a range of integers for x-axis to ensure equal spacing
    x_positions = range(len(shots))
    
    plt.rcParams.update({
        'font.size': 14,            # Default text size
        'axes.titlesize': 16,       # Title font size
        'axes.labelsize': 14,       # X and Y label size
        'xtick.labelsize': 12,      # X-tick label size
        'ytick.labelsize': 12,      # Y-tick label size
        'legend.fontsize': 12       # Legend font size
    })
    
    # Adjust the figure size to make the plot smaller
    plt.figure(figsize=(8, 5))
    
    for method, results in results_dict.items():
        avg_accuracies = [np.mean(results[shot]['auc']) for shot in shots]
        wrapped_method = method.replace('+ ', '+\n')
        plt.plot(x_positions, avg_accuracies, marker='o', label=wrapped_method)
    
    # Set the custom ticks and labels
    plt.xticks(ticks=x_positions, labels=shots)
    plt.xlabel('Number of Shots')
    plt.ylabel('AUC (%)')
    
    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend()
    plt.grid()
    # Save the plot as a PDF
    file_name = f'{data_name}_{ml_name}_{type}.pdf'
    if save:
        plt.savefig(os.path.join(save_path, file_name), format='pdf', bbox_inches='tight')
    
    plt.show()


def plot_results_with_shade(data_name, results_dict, shots, save_path='./plots'):
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Use a range of integers for x-axis to ensure equal spacing
    x_positions = range(len(shots))
    
    for method, results in results_dict.items():
        avg_accuracies = [np.mean(results[shot]['auc']) for shot in shots]
        std_devs = [np.std(results[shot]['auc']) for shot in shots]
        
        # Plot the average accuracies
        plt.plot(x_positions, avg_accuracies, marker='o', label=method)
        
        # Fill the area between the standard deviation bounds
        plt.fill_between(x_positions, 
                         [a - s for a, s in zip(avg_accuracies, std_devs)], 
                         [a + s for a, s in zip(avg_accuracies, std_devs)], 
                         alpha=0.2)
    
    # Set the custom ticks and labels
    plt.xticks(ticks=x_positions, labels=shots)
    plt.xlabel('Number of Shots')
    plt.ylabel('Average AUC (%)')
    
    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Enable gridlines
    plt.grid(True)
    
    # Save the plot as a PDF
    file_name = f'{data_name}_feature_selection_with_shade.pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf', bbox_inches='tight')
    
    plt.show()


def compute_p_value_for_auc(log_reg_results_pyramid, log_reg_results_llm_select, rf_results_pyramid, rf_results_llm_select, shots):
    # Combine the AUC values across all shots into vectors
    pyramid_auc_log_reg = []
    llm_select_auc_log_reg = []
    pyramid_auc_rf = []
    llm_select_auc_rf = []
    
    for shot in shots:
        # Logistic Regression AUC values
        pyramid_auc_log_reg.extend(log_reg_results_pyramid[shot]['auc'])
        llm_select_auc_log_reg.extend(log_reg_results_llm_select[shot]['auc'])
        
        # Random Forest AUC values
        pyramid_auc_rf.extend(rf_results_pyramid[shot]['auc'])
        llm_select_auc_rf.extend(rf_results_llm_select[shot]['auc'])
    
    # Perform paired two-tailed t-test for Logistic Regression
    t_stat_log_reg, p_value_log_reg = ttest_rel(pyramid_auc_log_reg, llm_select_auc_log_reg)
    
    # Perform paired two-tailed t-test for Random Forest
    t_stat_rf, p_value_rf = ttest_rel(pyramid_auc_rf, llm_select_auc_rf)
    
    # Print the p-values
    print(f"Logistic Regression AUC p-value: {p_value_log_reg:.8f}")
    print(f"Random Forest AUC p-value: {p_value_rf:.8f}")
    
    return p_value_log_reg, p_value_rf