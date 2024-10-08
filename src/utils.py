"""
DISCLAIMER:
This code is inspired from the open-source project FeatLLM developed by Sungwon Han.
You can find the original repository at: https://github.com/Sungwon-Han/FeatLLM.
"""
import warnings
import requests
import os
import random
from tqdm import tqdm
import pandas as pd
import re
from openai import OpenAI
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from config import OPENAI_API_KEY, TOGETHER_API_KEY
import time
import torch
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from together import Together
import json
import pandas as pd
import os
from tabulate import tabulate
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

TASK_DICT = {
    'ancestry_15_features': "What is the subject's genomic ancestry? European, South Asian, East Asian, African, or American?",
    'hearing_loss_15_features': "Does the subject have hereditary hearing loss? With regards to SNP variants, no mutations being found for the SNP are indicated by 0, heterozygous mutations by 1, and homozygous mutations by 2.",
}
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
client = OpenAI()
client_tog = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

"""
* Adjusted for SNP dataset
* 'y' column must come first in df
"""
def get_dataset(data_name, shot, seed, test_size_split = 0.5, preloaded_df=None):
    if preloaded_df is not None:
        df = preloaded_df
        
    else:
        file_name = f"../data/{data_name}.csv"
        df = pd.read_csv(file_name)

    df = df.dropna()

    if 'hearing_loss' in data_name:
        mask = (df != ' ').all(axis=1)
        df = df[mask]
        with pd.option_context('mode.chained_assignment', None):
            df.loc[:, df.columns != 'y'] = df.loc[:, df.columns != 'y'].astype(int)
            default_target_attribute = 'y'
    else: 
        if "aims_15" in data_name:
            default_target_attribute = 'y'
        else:
            default_target_attribute = 'superpopulation_name'
        with pd.option_context('mode.chained_assignment', None):
            df.loc[:, df.columns != default_target_attribute] = df.loc[:, df.columns != default_target_attribute].astype(int)
            df = df.astype({col: 'int' for col in df.select_dtypes(include=['float64']).columns})
        
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][1:]
    attribute_names = df.columns[1:].tolist()

    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()
    label_list = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(default_target_attribute, axis=1),
        y,
        test_size=test_size_split,
        random_state=seed,
        stratify=y
    )
    
    # assert(shot <= 128) # We only consider the low-shot regimes here
    X_new_train = X_train.copy()
    X_new_train[default_target_attribute] = y_train
    sampled_list = []
    total_shot_count = 0
    remainder = shot % len(np.unique(y_train))
    for _, grouped in X_new_train.groupby(default_target_attribute):
        sample_num = shot // len(np.unique(y_train))
        if remainder > 0:
            sample_num += 1
            remainder -= 1
        grouped = grouped.sample(sample_num, random_state=seed)
        sampled_list.append(grouped)
    X_balanced = pd.concat(sampled_list)
    X_train = X_balanced.drop([default_target_attribute], axis=1)
    y_train = X_balanced[default_target_attribute].to_numpy()

    return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator

"""Added multiclass analysis.
"""

def evaluate(pred_probs, answers, multiclass=False, class_level_analysis=False, label_list=None):
    if not multiclass:
        result_auc = roc_auc_score(answers, pred_probs[:, 1])
        pred_labels = np.argmax(pred_probs, axis=1)
        result_accuracy = accuracy_score(answers, pred_labels)
        result_f1 = f1_score(answers, pred_labels, average='weighted')
    else:
        result_auc = roc_auc_score(answers, pred_probs, multi_class='ovr', average='macro')
        pred_labels = np.argmax(pred_probs, axis=1)
        result_accuracy = accuracy_score(answers, pred_labels)
        result_f1 = f1_score(answers, pred_labels, average='weighted')
    
    metrics = {
        'AUC': result_auc,
        'Accuracy': result_accuracy,
        'F1-Score': result_f1
    }
    
    if class_level_analysis:
        precision, recall, f1, _ = precision_recall_fscore_support(answers, pred_labels, average=None)
        for i, class_label in enumerate(np.unique(answers)):
            metrics[f'Class {label_list[class_label]} Precision'] = precision[i]
            metrics[f'Class {label_list[class_label]} Recall'] = recall[i]
            metrics[f'Class {label_list[class_label]} F1-Score'] = f1[i]

    return metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def query_gpt(text_list, max_tokens=100, temperature=0, max_try_num=10, model="gpt-3.5-turbo"):
    result_list = []
    for prompt in tqdm(text_list):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                if 'gpt' not in model and 'o1' not in model:
                    response = client_tog.chat.completions.create(
                        model='meta-llama/' + model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.7,
                        top_k=50,
                        repetition_penalty=1,
                        stop=["<|eot_id|>"]
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                    )   
                result = response.choices[0].message.content.strip()
                result_list.append(result)
                break
            except Exception as e:
                if 'gpt' in model:
                    print(f"Error making OpenAI API call: {e}")
                else: 
                    print(f"Error making LLM API call: {e}")
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                time.sleep(10)
    return result_list

def query_full_gpt(text_list, max_tokens=30, temperature=0, max_try_num=10, model="gpt-3.5-turbo"):
    result_list = []
    for prompt in tqdm(text_list):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt['system']},
                        {"role": "user", "content": prompt['user']}
                    ],
                    temperature=temperature
                )

                result = response.choices[0].message.content.strip()
                result_list.append(result)
                break
            except Exception as e:
                if 'gpt' in model:
                    print(f"Error making OpenAI API call: {e}")
                else: 
                    print(f"Error making LLM API call: {e}")
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                time.sleep(10)
    return result_list

def serialize(row, prompt_version = "v4"):
    target_str = f""
    for attr_idx, attr_name in enumerate(list(row.index)):
        if attr_idx < len(list(row.index)) - 1:
            if int(prompt_version[1]) == 6:
                target_str += " of the person has ".join(["The genetic variant " + attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
                target_str += " minor alleles. "
            else:
                target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
                target_str += ". "
        else:
            if len(attr_name.strip()) < 2:
                continue
            if int(prompt_version[1]) == 6:
                target_str += " of the person has ".join(["The genetic variant " + attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
                target_str += " minor alleles. "
            else: 
                target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
                target_str += "."
    return target_str


def fill_in_templates(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str    

def extract_ancestry(text):
    # Define potential ancestry types for better matching
    possible_ancestries = ["African Ancestry", "South Asian Ancestry", "East Asian Ancestry", "American Ancestry", "European Ancestry"]
    
    # Regular expression pattern to find any ancestry type
    patterns = [re.compile(rf'{ancestry}', re.IGNORECASE) for ancestry in possible_ancestries]

    # Iterate over patterns and find matches
    for pattern in patterns:
        matches = pattern.findall(text)
        if matches:
            return matches[0].strip()

    # If no matches are found, return None
    return None

def extract_snp(text):
    # Use regex to find all instances of SNPs that start with "rs"
    snp_pattern = r'rs\d+'
    snps = re.findall(snp_pattern, text)
    
    # Get the last 15 SNPs
    last_fifteen_snps = snps[-15:]
    
    return last_fifteen_snps

def parse_rules_simple(rule_string):
    # Split the string by newline and dash, and strip any leading/trailing whitespace
    rules_list = []
    for rule_string in rule_string:
        rules = rule_string.split('\n') 
        for i,rule in enumerate(rules):
            rules[i] = rule[1:].strip(' `')
        rules_list.append(rules)
    return rules_list

    
def parse_rules(result_texts, label_list=[], truncation_index = None, interactions = False):
    total_rules = []
    for i, text in enumerate(result_texts):
        splitter = "onditions for class"
        if interactions:
            splitter = "nteractions for class"
        # print('--------raw-----------')
        # print(text)
        # print('------raw-------')
        splitted = re.split(f'(?i){splitter}', text)
        # print('--------after splitting-------')
        # print(splitted)
        # print('--------after splitting-------')
        if splitter not in text.lower():
            splitter = "####"
            splitted = re.split(f'(?i){splitter}', text)
            print(f"Query {i}: using #### as splitter")
        if splitter not in text.lower():
            splitter = "###"
            splitted = re.split(f'(?i){splitter}', text)
            print(f"Query {i}: using ### as splitter")
        if splitter not in text.lower():
            splitter = "##"
            splitted = re.split(f'(?i){splitter}', text)
            print(f"Query {i}: using ## as splitter")
        if splitter not in text.lower():
            splitter = r"\n**"
            splitted = re.split(f'(?i){splitter}', text)
            splitter = "\n**"
            print(f"Query {i}: using ** as splitter")
        if splitter not in text.lower():
            print(f"Skipping query {i} in parsing rules because splitter doesn't exist")
            continue
        if truncation_index is None and len(label_list) != 0 and len(splitted) != len(label_list) + 1:
            print(f"Skipping query {i} in parsing rules because {len(splitted)}")
            continue
        rule_raws = splitted[-len(label_list):]
        if truncation_index is not None:
            rule_raws = splitted[1:truncation_index]
            
        rule_dict = {}
        for rule_raw in rule_raws:
            # print(f'{i}:',rule_raw)
            if ':' in rule_raw: 
                class_name = rule_raw.split(":")[0].strip(" .'").strip(' []"')
            else: 
                class_name = extract_ancestry(rule_raw)
            rule_parsed = []
            for txt in rule_raw.strip().split("\n")[1:]:
                if len(rule_parsed) > 0 and len(txt) < 2:
                    break
                rule_parsed.append(" ".join(txt.strip().split(" ")[1:]))
                rule_dict[class_name] = rule_parsed
        total_rules.append(rule_dict)
        # print(f'{i}: ',rule_dict)
    return total_rules

def get_prompt_for_asking(data_name, df_all, df_x, df_y, label_list, 
                          default_target_attribute, file_name, meta_file_name, is_cat, num_query=5, num_conditions= 10, interactions=False, mixed = False, prompt_version="v4"):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()
    task_desc = f"{TASK_DICT[data_name]}\n"    
    
    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except:
        meta_data = {}
    print(meta_data)
    task_desc = f"{TASK_DICT[data_name]}\n"    
    df_incontext = df_x.copy()
    df_incontext[default_target_attribute] = df_y 
    
    format_list = [f'{num_conditions} different conditions for class "{label}":\n- [Condition]\n...' for label in label_list]
    if interactions:
        format_list = [f'{num_conditions} different feature interactions for class "{label}":\n- [Feature Interaction]\n...' for label in label_list]
    elif mixed:
        format_list = [f'{num_conditions} different conditions or feature interactions for class "{label}":\n- [Condition] or [Feature Interaction]\n...' for label in label_list]
    format_desc = '\n\n'.join(format_list)
            
    template_list = []
    current_query_num = 0
    end_flag = False
    while True:     
        if current_query_num >= num_query:
            break
                        
        # Feature bagging
        if len(df_incontext.columns) >= 20:
            total_column_list = []
            for i in range(len(df_incontext.columns) // 10):
                column_list = df_incontext.columns.tolist()[:-1]
                random.shuffle(column_list)
                total_column_list.append(column_list[i*10:(i+1)*10])
        else:
            total_column_list = [df_incontext.columns.tolist()[:-1]]
            
        for selected_column in total_column_list:
            if current_query_num >= num_query:
                break
                
            # Sample bagging
            threshold = 16   
            if len(df_incontext) > threshold:
                sample_num = int(threshold / df_incontext[default_target_attribute].nunique())
                df_incontext = df_incontext.groupby(
                    default_target_attribute, group_keys=False
                ).apply(lambda x: x.sample(sample_num))
                
            feature_name_list = []
            sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column]
            is_cat_sel = np.array(is_cat)[sel_cat_idx]
            
            for cidx, cname in enumerate(selected_column):
                desc = meta_data[cname] if cname in meta_data.keys() else ""
                if 'hearing_loss' or 'aims' or 'ancestry' in data_name:
                    if int(prompt_version[1]) <= 4:
                        feature_name_list.append(f"- {cname}: {desc} (numerical variable with categories [0,1,2])")
                    elif desc == '':
                        feature_name_list.append(f"- {cname}")
                    else:
                        feature_name_list.append(f"- {cname}: {desc}")
                else:
                    feature_name_list.append(f"- {cname}: {desc} (numerical variable)")

            feature_desc = "\n".join(feature_name_list)
            
            in_context_desc = ""  
            df_current = df_incontext.copy()
            df_current = df_current.groupby(
                default_target_attribute, group_keys=False
            ).apply(lambda x: x.sample(frac=1))

            for icl_idx, icl_row in df_current.iterrows():
                answer = icl_row[default_target_attribute]
                icl_row = icl_row.drop(labels=default_target_attribute)  
                icl_row = icl_row[selected_column]
                in_context_desc += serialize(icl_row, prompt_version=prompt_version)
                if 'hearing_loss' in data_name:
                    if answer == "Yes":
                        in_context_desc += f"\nAnswer: {answer}, has hereditary hearing loss\n\n"
                    else:
                        in_context_desc += f"\nAnswer: {answer}, does not have hereditary hearing loss\n\n"
                else:
                    in_context_desc += f"\nAnswer: {answer}\n\n"

            fill_in_dict = {
                "[TASK]": task_desc, 
                "[EXAMPLES]": in_context_desc,
                "[FEATURES]": feature_desc,
                "[FORMAT]": format_desc,
                "[NUM_FEATURES]": str(num_conditions)
            }
            template = fill_in_templates(fill_in_dict, prompt_type_str)
            template_list.append(template)
            current_query_num += 1
        
    return template_list, feature_desc

def get_prompt_for_generating_function(parsed_rule, feature_desc, file_name):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()
    
    template_list = []
    function_name = f'extracting_engineered_features'

    fill_in_dict = {
        "[NAME]": function_name, 
        "[CONDITIONS]": parsed_rule,
        "[FEATURES]": feature_desc
    }
    template = fill_in_templates(fill_in_dict, prompt_type_str)
    template_list.append(template)
        
    return template_list


def convert_to_binary_vectors_simple(fct_strs_all, fct_names, X_train, X_test, num_of_features = 10, strict_num_of_features = False, include_original_features= False):
    X_train_all_dict = {}
    X_test_all_dict = {}
    executable_list = [] # Save the parsed functions that are properly working for both train/test sets
    for i in range(len(fct_strs_all)): # len(fct_strs_all) == # of trials for ensemble
        try:
            # Diagnostic print statements to pinpoint which function is failing
            exec(fct_strs_all[i].strip('` "'))
            X_train_each = locals()[fct_names[i]](X_train).astype('int').to_numpy()
            X_test_each = locals()[fct_names[i]](X_test).astype('int').to_numpy()
            assert(X_train_each.shape[1] == X_test_each.shape[1])
            assert(X_train_each.shape[0] == X_train.shape[0])
            if strict_num_of_features:
                assert(X_train_each.shape[1] == num_of_features)
            X_train_dict = torch.tensor(X_train_each).float()
            X_test_dict = torch.tensor(X_test_each).float()
            if include_original_features:
                X_train_all_dict[i] = torch.cat([torch.tensor(X_train.astype('float32').to_numpy()).float(), X_train_dict], dim=1)
                X_test_all_dict[i] = torch.cat([torch.tensor(X_test.astype('float32').to_numpy()).float(), X_test_dict], dim=1)
            else:
                X_train_all_dict[i] = X_train_dict
                X_test_all_dict[i] = X_test_dict
            executable_list.append(i)
        except Exception as e: # If error occurred during the function call, remove the current trial
            try:
                print(f"""Iteration {i}, Error in convert_to_binary_vectors: {e}
                    Is the # of columns in X_train equal to X_test after applying the functions? {X_train_each.shape[1] == X_test_each.shape[1]}
                    Is the # of rows the same after applying the functions? {X_train_each.shape[0] == X_train.shape[0]}
                    How many conditional features after applying the functions? {X_train_each.shape[1]}""")
                continue
            except Exception:
                continue
            
    return executable_list, X_train_all_dict, X_test_all_dict

def combine_parsed_rules(list1, list2):
    combined_list = []
    # Check if both lists have the same length
    if len(list1) != len(list2):
        warnings.warn("Both parsed lists should have the same length")
        min_length = min(len(list1), len(list2))
        list1 = list1[:min_length]
        list2 = list2[:min_length]
    for dict1, dict2 in zip(list1, list2):
        combined_dict = {}
        # Iterate over keys in the first dictionary
        for key in dict1:
            if key in dict2:
                combined_dict[key] = dict1[key] + dict2[key]
            else:
                combined_dict[key] = dict1[key]
        # Add remaining keys from the second dictionary
        for key in dict2:
            if key not in combined_dict:
                combined_dict[key] = dict2[key]
        combined_list.append(combined_dict)
        
    return combined_list

def diagnose(fct, X_train, num_conditions, condition_tolerance = 2):
    try:
        fct_name = fct.strip('` "').split('def')[1].split('(')[0].strip()
        exec(fct.strip('` "'))
        # Being off by 1 is fine
        if abs(len(locals()[fct_name](X_train).astype('int').to_numpy()[0]) - num_conditions) > condition_tolerance:
            raise Exception("The number of conditions does not match.")
        else:
            return False, "all good"
    except Exception as e:
        print(e)
        return True, e

def self_critique_functions(parsed_rules, feature_desc, fct_strs_all, X_train, _NUM_OF_CONDITIONS, _NUM_OF_CONDITIONS_FOR_INTERACTIONS, _REWRITING_FUNCTION_MODEL, condition_tolerance=2):    
    critique_fct_strs_all = []
    for i, fct_strs in enumerate(fct_strs_all):
        has_error, error_message = diagnose(fct_strs, X_train, _NUM_OF_CONDITIONS + _NUM_OF_CONDITIONS_FOR_INTERACTIONS, condition_tolerance=condition_tolerance)
        if has_error:
            print("Function string to critique:", fct_strs)  # Debugging statement
            instructions_for_query_i = get_prompt_for_generating_function(parsed_rules[i], 
                                                                    feature_desc,
                                                                    './templates/ask_for_function.txt')
            context = f"""The function is supposed to follow these instructions:\n\n'''{instructions_for_query_i}'''\n\nHowever, the following function was flagged for the following error: '{error_message}'! Fix this error and output just the fixed function. Wrap only the function part with <start> and <end>, and do not add any comments, descriptions, and package importing lines in the code. Here's the function you need to fix:\n\n"""
            critique_results = query_gpt([context + fct_strs], max_tokens=2500, temperature=0, model=_REWRITING_FUNCTION_MODEL)
            critique_fct_strs_all.append(next(iter(critique_results)).split('<start>')[1].split('<end>')[0].strip())
        else:
            critique_fct_strs_all.append(fct_strs)
    return critique_fct_strs_all

def evaluate_models_on_transformed_data(X_train_all_dict, X_test_all_dict, y_train_num, y_test_num, label_list, multiclass=False):
    # Initialize dictionaries to store models for each method
    lr_models = []
    rf_models = []
    xgb_models = []
    first_key = list(X_train_all_dict.keys())[0]
    # Initialize arrays to store ensemble predictions for each model type
    ensemble_pred_probs_train_lr = np.zeros((X_train_all_dict[first_key].shape[0], len(label_list)))
    ensemble_pred_probs_test_lr = np.zeros((X_test_all_dict[first_key].shape[0], len(label_list)))

    ensemble_pred_probs_train_rf = np.zeros((X_train_all_dict[first_key].shape[0], len(label_list)))
    ensemble_pred_probs_test_rf = np.zeros((X_test_all_dict[first_key].shape[0], len(label_list)))

    ensemble_pred_probs_train_xgb = np.zeros((X_train_all_dict[first_key].shape[0], len(label_list)))
    ensemble_pred_probs_test_xgb = np.zeros((X_test_all_dict[first_key].shape[0], len(label_list)))
    num_of_conditions = []
    # Train LR, RF, XGBoost on each version of the training data
    for i, (X_train_now, X_test_now) in enumerate(zip(X_train_all_dict.values(), X_test_all_dict.values())):
        print(f"# of Conditions in query {i}: ",X_train_now.shape[1])
        num_of_conditions.append(X_train_now.shape[1])
        # Train Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_now, y_train_num)
        lr_models.append(lr)
        lr_pred_probs_train = lr.predict_proba(X_train_now)
        lr_pred_probs_test = lr.predict_proba(X_test_now)

        ensemble_pred_probs_train_lr += lr_pred_probs_train
        ensemble_pred_probs_test_lr += lr_pred_probs_test
        
        # Train Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train_now, y_train_num)
        rf_models.append(rf)
        rf_pred_probs_train = rf.predict_proba(X_train_now)
        rf_pred_probs_test = rf.predict_proba(X_test_now)

        ensemble_pred_probs_train_rf += rf_pred_probs_train
        ensemble_pred_probs_test_rf += rf_pred_probs_test
        
        # Train XGBoost
        xgb = XGBClassifier( eval_metric='mlogloss')
        xgb.fit(X_train_now, y_train_num)
        xgb_models.append(xgb)
        xgb_pred_probs_train = xgb.predict_proba(X_train_now)
        xgb_pred_probs_test = xgb.predict_proba(X_test_now)

        ensemble_pred_probs_train_xgb += xgb_pred_probs_train
        ensemble_pred_probs_test_xgb += xgb_pred_probs_test

    print("Average number of conditions in ensembled model: ", np.mean(num_of_conditions))
    # Average the probabilities for each model type
    ensemble_pred_probs_train_lr /= len(X_train_all_dict)
    ensemble_pred_probs_test_lr /= len(X_test_all_dict)

    ensemble_pred_probs_train_rf /= len(X_train_all_dict)
    ensemble_pred_probs_test_rf /= len(X_test_all_dict)

    ensemble_pred_probs_train_xgb /= len(X_train_all_dict)
    ensemble_pred_probs_test_xgb /= len(X_test_all_dict)

    # Evaluate the ensemble predictions
    ensemble_metrics_train_lr = evaluate(
        ensemble_pred_probs_train_lr, 
        y_train_num, 
        multiclass=multiclass, 
        class_level_analysis=True, 
        label_list=label_list
    )

    ensemble_metrics_test_lr = evaluate(
        ensemble_pred_probs_test_lr, 
        y_test_num, 
        multiclass=multiclass, 
        class_level_analysis=True, 
        label_list=label_list
    )

    ensemble_metrics_train_rf = evaluate(
        ensemble_pred_probs_train_rf, 
        y_train_num, 
        multiclass=multiclass, 
        class_level_analysis=True, 
        label_list=label_list
    )

    ensemble_metrics_test_rf = evaluate(
        ensemble_pred_probs_test_rf, 
        y_test_num, 
        multiclass=multiclass, 
        class_level_analysis=True, 
        label_list=label_list
    )

    ensemble_metrics_train_xgb = evaluate(
        ensemble_pred_probs_train_xgb, 
        y_train_num, 
        multiclass=multiclass, 
        class_level_analysis=True, 
        label_list=label_list
    )

    ensemble_metrics_test_xgb = evaluate(
        ensemble_pred_probs_test_xgb, 
        y_test_num, 
        multiclass=multiclass, 
        class_level_analysis=True, 
        label_list=label_list
    )

    # Structure the results for returning
    results = {
        "lr": {
            "train": ensemble_metrics_train_lr,
            "test": ensemble_metrics_test_lr,
        },
        "rf": {
            "train": ensemble_metrics_train_rf,
            "test": ensemble_metrics_test_rf,
        },
        "xgboost": {
            "train": ensemble_metrics_train_xgb,
            "test": ensemble_metrics_test_xgb,
        }
    }

    return results


def evaluate_our_method(_DATA, _NUM_QUERY,_SHOT,_SEEDS, _MODEL,_FUNCTION_MODEL, _PROMPT_VERSION, _METADATA_VERSION, _NOTE="", _RECORD_LOGS=True, condition_tolerance=10, include_original_features = False):
    # Parameterization & Initialization
    evaluation_results = {'lr': {'train': {'accuracy': [], 'f1': [], 'auc': []},
                                 'test': {'accuracy': [], 'f1': [], 'auc': []}},
                          'rf': {'train': {'accuracy': [], 'f1': [], 'auc': []},
                                 'test': {'accuracy': [], 'f1': [], 'auc': []}},
                          'xgboost': {'train': {'accuracy': [], 'f1': [], 'auc': []},
                                      'test': {'accuracy': [], 'f1': [], 'auc': []}}}
    _REWRITING_FUNCTION_MODEL = "gpt-4-1106-preview"
    if "gpt" in _MODEL:
        _LLM_FILE = 'ask_llm'
        _MAX_TOKENS = 2500
    else: 
        _LLM_FILE = 'ask_llm_llama'
        _MAX_TOKENS = 2000
    if 'ancestry' in _DATA:
        _DATA_TYPE = 'ancestry'
    else:
        _DATA_TYPE = 'hearing_loss'
    for seed in _SEEDS:
        print(f"------------------------Iteration for Seed {seed}------------------------")

        # Set the random seed
        set_seed(seed)
        
        # Assume get_dataset retrieves your data according to the shot and seed
        df, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = get_dataset(_DATA, _SHOT, seed)
        X_all = df.drop(target_attr, axis=1)
        print(f"------------------------Generating Prompt Template------------------------")

        ask_file_name = f'./templates/{_LLM_FILE}_{_PROMPT_VERSION}_{_DATA_TYPE}.txt'
        meta_data_name = f"../data/{_DATA}-metadata-{_METADATA_VERSION}.json"
        templates, feature_desc = get_prompt_for_asking(
            _DATA, X_all, X_train, y_train, label_list, target_attr, ask_file_name, 
            meta_data_name, is_cat, num_query=_NUM_QUERY, num_conditions=15,
            prompt_version =_PROMPT_VERSION 
        )
        # We make sure all printing goes to an .out file

        print(templates[0])
        
        _DIVIDER = "\n\n---DIVIDER---\n\n"
        _VERSION = "\n\n---VERSION---\n\n"
        print(f"------------------------Generating Rules------------------------")
        rule_file_name = f'./rules/{_DATA}/{_SHOT}_shot/rule-s{_SHOT}-{_MODEL}-{_PROMPT_VERSION}-q{_NUM_QUERY}-{seed}{_NOTE}.out'
        if os.path.isfile(rule_file_name) == False:
            results = query_gpt(templates, max_tokens=_MAX_TOKENS, temperature=1, model = _MODEL)
            if _RECORD_LOGS:
                with open(rule_file_name, 'w') as f:
                    total_rules = _DIVIDER.join(results)
                    f.write(total_rules)
        else:
            with open(rule_file_name, 'r') as f:
                total_rules_str = f.read().strip()
                results = total_rules_str.split(_DIVIDER)

        print(results[0])
        
        skip_critique = False
        print(f"------------------------Extracting Rules------------------------")
        saved_file_name = f'./rules/{_DATA}/{_SHOT}_shot/function-s{_SHOT}-{_MODEL}-{_FUNCTION_MODEL}-{_PROMPT_VERSION}-q{_NUM_QUERY}-{seed}{_NOTE}.out'    
        # If we don't have this in function form, 
        if os.path.isfile(saved_file_name) == False:   
            parsed_rules = []

            # Iterate through each result in the results list
            for result in results:
                # Use query_gpt to transform each result
                transformed_result = query_gpt(
                    [f"Extract the list of engineered features (include their equation or instructions) and list them one after another in a new line: {result}\n\nIf some features are clumped up together, make sure to list them separately.\n\nList:"], 
                    max_tokens=_MAX_TOKENS, 
                    temperature=0, 
                    model=_FUNCTION_MODEL
                )
                # Append the transformed result to the results_transformed list
                parsed_rules.append(transformed_result[0])

            # The parsed_rules list now contains all the transformed results
            print(parsed_rules[0])
        else: 
            skip_critique = True         



        print(f"------------------------Generating Functions from Rules------------------------")

        saved_file_name = f'./rules/{_DATA}/{_SHOT}_shot/function-s{_SHOT}-{_MODEL}-{_FUNCTION_MODEL}-{_PROMPT_VERSION}-q{_NUM_QUERY}-{seed}{_NOTE}.out'    
        if os.path.isfile(saved_file_name) == False:
            function_file_name = './templates/ask_for_function_v2.txt'
            fct_strs_all = []
            for parsed_rule in tqdm(parsed_rules):
                print(parsed_rule)
                fct_template = get_prompt_for_generating_function(
                    parsed_rule, feature_desc, function_file_name
                )
                print(fct_template)
                fct_results = query_gpt(fct_template, max_tokens=2500, temperature=0, model = _FUNCTION_MODEL)
                print(fct_results)
                fct_strs = [fct_txt.split('<start>')[1].split('<end>')[0].strip() for fct_txt in fct_results]
                print(fct_strs)
                fct_strs_all.append(fct_strs[0])
        else:
            with open(saved_file_name, 'r') as f:
                total_str = f.read().strip()
                fct_strs_all = [x for x in total_str.split(_VERSION)]
        
        print(f"------------------------Self-Fix Function------------------------")
        if not skip_critique:
            critique_fct_strs_all = self_critique_functions(parsed_rules, feature_desc, fct_strs_all, X_train, 15, 5, _REWRITING_FUNCTION_MODEL, condition_tolerance=condition_tolerance)
            if _RECORD_LOGS:
                with open(saved_file_name, 'w') as f:
                    total_str = _VERSION.join([x for x in critique_fct_strs_all])
                    f.write(total_str)
        else: 
            critique_fct_strs_all = fct_strs_all
            
                
        # Get function names and strings
        fct_names = []
        fct_strs_final = []
        for fct_str in critique_fct_strs_all:
            if 'def' not in fct_str:
                continue
            fct_names.append(fct_str.split('def')[1].split('(')[0].strip())
            fct_strs_final.append(fct_str)
                
        mask = X_test.notna().all(axis=1)
        # Dropping weird NAs
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        print(f"------------------------Evaluating Downstream Performance------------------------")

        executable_list, X_train_all_dict, X_test_all_dict = convert_to_binary_vectors_simple(fct_strs_final, 
                                                                                     fct_names, 
                                                                                     X_train, 
                                                                                     X_test, 
                                                                                     num_of_features=20,
                                                                                     include_original_features=include_original_features)
        

        print(len(executable_list))

        multiclass = True if len(label_list) > 2 else False
        y_train_num = np.array([label_list.index(k) for k in y_train])
        y_test_num = np.array([label_list.index(k) for k in y_test])
        
        seed_results = evaluate_models_on_transformed_data(
            X_train_all_dict=X_train_all_dict,
            X_test_all_dict=X_test_all_dict,
            y_train_num=y_train_num,
            y_test_num=y_test_num,
            label_list=label_list,
            multiclass=multiclass
        )
        
        # Collect results for each model type (lr, rf, xgboost)
        for model_type in evaluation_results.keys():
            for phase in ['train', 'test']:
                evaluation_results[model_type][phase]['accuracy'].append(seed_results[model_type][phase]['Accuracy'])
                evaluation_results[model_type][phase]['f1'].append(seed_results[model_type][phase]['F1-Score'])
                evaluation_results[model_type][phase]['auc'].append(seed_results[model_type][phase]['AUC'])

                # Append per-class F1 scores as well
                for ancestry in label_list:
                    key = f'Class {ancestry} F1-Score'
                    if key not in evaluation_results[model_type][phase]:
                        evaluation_results[model_type][phase][key] = []
                    evaluation_results[model_type][phase][key].append(seed_results[model_type][phase][key])

    # Save the results to a pickle file
    
    file_path = f"logs/{_DATA}/{_SHOT}_shot_{_MODEL}_{_FUNCTION_MODEL}_{_PROMPT_VERSION}_q{_NUM_QUERY}_{'_'.join(map(str, _SEEDS))}{_NOTE}.pkl"
    if include_original_features:
            file_path = f"logs/{_DATA}/{_SHOT}_shot_{_MODEL}_{_FUNCTION_MODEL}_{_PROMPT_VERSION}_q{_NUM_QUERY}_{'_'.join(map(str, _SEEDS))}{_NOTE}_plus_original.pkl"
    
    if _RECORD_LOGS:
        with open(file_path, 'wb') as f:
            pickle.dump(evaluation_results, f)
        
    print(f"Results saved to {file_path}")
    print_seed_results(evaluation_results)
    return evaluation_results

def print_seed_results(data):
    for model, phases in data.items():
        print(f"{model.upper()} MODEL RESULTS")
        print("-" * 40)
        for phase, metrics in phases.items():
            print(f"{phase.capitalize()} Results:")
            print("-" * 30)
            for metric, values in metrics.items():
                print(f"{metric}: {values}")
            print("-" * 30)
        print("=" * 40)
        
# This is printing the results for FREEFORM feature engineering
    """Input: Filepaths for each shot .pkl file and then we combine and turn into a format ready for printing or plotting
    """
def load_and_print_results(file_paths):
    table = []
    geno_llm_lr_results_ancestry, geno_llm_rf_results_ancestry, geno_llm_xgb_results_ancestry = {}, {}, {}
    
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract shot size from the file name (assumes the format "logs/.../10_shot_...pkl")
        shot = int(os.path.basename(file_path).split('_')[0])
        geno_llm_lr_results_ancestry[shot] = results['lr']['test']
        geno_llm_rf_results_ancestry[shot] = results['rf']['test']
        geno_llm_xgb_results_ancestry[shot] = results['xgboost']['test']
        
        # Aggregate metrics for each model type and phase (train, test)
        for model_type in results.keys():
            # Calculate averages and standard errors
            avg_train_accuracy = np.mean(results[model_type]['train']['accuracy'])
            std_train_accuracy = np.std(results[model_type]['train']['accuracy']) 
            avg_train_f1 = np.mean(results[model_type]['train']['f1'])
            std_train_f1 = np.std(results[model_type]['train']['f1']) 
            avg_train_auc = np.mean(results[model_type]['train']['auc'])
            std_train_auc = np.std(results[model_type]['train']['auc'])

            avg_test_accuracy = np.mean(results[model_type]['test']['accuracy'])
            std_test_accuracy = np.std(results[model_type]['test']['accuracy']) 
            avg_test_f1 = np.mean(results[model_type]['test']['f1'])
            std_test_f1 = np.std(results[model_type]['test']['f1']) 
            avg_test_auc = np.mean(results[model_type]['test']['auc'])
            std_test_auc = np.std(results[model_type]['test']['auc']) 

            table.append([
                shot, model_type,
                f"{avg_train_accuracy:.4f}", f"{std_train_accuracy:.2f}",
                f"{avg_train_f1:.4f}", f"{std_train_f1:.2f}",
                f"{avg_train_auc:.4f}", f"{std_train_auc:.2f}",
                f"{avg_test_accuracy:.4f}", f"{std_test_accuracy:.2f}",
                f"{avg_test_f1:.4f}", f"{std_test_f1:.2f}",
                f"{avg_test_auc:.4f}", f"{std_test_auc:.2f}"
            ])
    
    headers = [
        "Shot", "Model", 
        "Train Acc", "Std", "Train F1", "Std", "Train AUC", "Std",
        "Test Acc", "Std", "Test F1", "Std", "Test AUC", "Std"
    ]
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    return geno_llm_lr_results_ancestry, geno_llm_rf_results_ancestry, geno_llm_xgb_results_ancestry

def print_average_f1_scores(file_paths):
    all_results = {}

    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract shot size from the file name
        shot = int(os.path.basename(file_path).split('_')[0])
        if shot not in all_results:
            all_results[shot] = {'train': {}, 'test': {}}

        # Aggregate F1-scores for each ancestry and each phase (train, test)
        for model_key in results.keys():
            for phase in ['train', 'test']:
                for key in results[model_key][phase].keys():
                    if 'F1-Score' in key and key != 'F1-Score':
                        ancestry = key.split()[-2]
                        if ancestry not in all_results[shot][phase]:
                            all_results[shot][phase][ancestry] = []
                        all_results[shot][phase][ancestry].append(results[model_key][phase][key])

    # Print the average F1-scores for each ancestry for both train and test phases
    table = []
    for shot, phases in sorted(all_results.items()):
        for phase, ancestries in phases.items():
            row = [shot, phase]
            for ancestry in sorted(ancestries.keys()):
                avg_f1 = np.mean(ancestries[ancestry])
                row.append(f"{avg_f1:.4f}")
            table.append(row)
    
    headers = ["Shot", "Phase"] + sorted(list(all_results[next(iter(all_results))]['train'].keys()))
    print(tabulate(table, headers=headers, tablefmt="pretty"))


def plot_f1_scores_across_shots(file_paths):
    all_results = {}

    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract shot size from the file name
        shot = int(os.path.basename(file_path).split('_')[0])
        if shot not in all_results:
            all_results[shot] = {'train': {}, 'test': {}}

        # Aggregate F1-scores for each ancestry and each phase (train, test)
        for model_key in results.keys():
            for phase in ['train', 'test']:
                for key in results[model_key][phase].keys():
                    if 'F1-Score' in key and key != 'F1-Score':
                        ancestry = key.split()[-2]
                        if ancestry not in all_results[shot][phase]:
                            all_results[shot][phase][ancestry] = []
                        all_results[shot][phase][ancestry].append(results[model_key][phase][key])

    # Plot the average F1-scores for each ancestry across shots
    plt.figure(figsize=(10, 6))
    
    for phase in ['train', 'test']:
        ancestries = sorted(list(all_results[next(iter(all_results))][phase].keys()))
        for ancestry in ancestries:
            f1_scores = [np.mean(all_results[shot][phase][ancestry]) for shot in sorted(all_results.keys())]
            plt.plot(sorted(all_results.keys()), f1_scores, marker='o', label=f"{phase.capitalize()} {ancestry} F1-Score")
    
    plt.xlabel('Shot Size')
    plt.ylabel('F1-Score')
    plt.title('Average F1-Scores for Each Ancestry Across Shots')
    plt.legend()
    plt.grid(True)
    plt.show()

###### LEGACY FUNCTIONS ########

def get_snp_gene(rsid):
    try:
        url = f'https://clinicaltables.nlm.nih.gov/api/snps/v3/search?terms={rsid}&df=rsNum,38.gene'
        response = requests.get(url).json()
        return response[3]
    except Exception as e:
        print(f"Error processing SNP {rsid}: {e}")
        return None, None, None

def get_snp_info(rsid):
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&id={rsid}&api_key=c478ddfa81b3c3317bd5d31e928620ca9708&retmode=json"
        response = requests.get(url).json()
        gene_name = response['result'][rsid]['genes'][0]['name'] if 'genes' in response['result'][rsid] and len(response['result'][rsid]['genes']) > 0 else None
        func_annot = response['result'][rsid].get('fxn_class', None)
        chrpos = response['result'][rsid].get('chrpos', None)
        return gene_name, func_annot, chrpos
    except Exception as e:
        print(f"Error processing SNP {rsid}: {e}")
        return None, None, None
    
def gene_to_json_line(gene):
    if ' ' in gene: 
        genes = gene.split(' ')
        return f' is a variant associated with the {genes[0]} gene and {genes[1]} gene'
    elif gene == '':
        return ' is located within a non-coding region'
    else:
        return f' is a variant associated with the {gene} gene'

def retrieve_SNP_and_generate_json(_DATA, target_attr = 'y'):
    df = pd.read_csv(f'../data/{_DATA}.csv')
    df_filtered = df.drop(columns=[target_attr])
    data_dict = {col: col + gene_to_json_line(get_snp_gene(col)[0][1]) for col in df_filtered.columns}
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data_dict, indent=4)
    # Save to a .json file
    with open(f'../data/{_DATA}.json', 'w') as f:
        f.write(json_data)

def generate_empty_json(_DATA, target_attr = 'y'):
    df = pd.read_csv(f'../data/{_DATA}.csv')
    df_filtered = df.drop(columns=[target_attr])
    data_dict = {col: '' for col in df_filtered.columns}
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data_dict, indent=4)
    # Save to a .json file
    with open(f'../data/{_DATA}.json', 'w') as f:
        f.write(json_data)
        
def clean_rsid_csv(_DATA):
    df = pd.read_csv(f'../data/{_DATA}.csv')
    df = df.drop(columns=['sample', 'biosample','PC1','PC2','sex','biosample','population','superpopulation','superpopulation_name'])
    df = df.rename(columns = {'population_name':'y'})
    # Remove the 'y' column
    df_filtered = df.drop(columns=['y'])
    data_dict = {col: "" for col in df_filtered.columns}
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data_dict, indent=4)
    # Save to a .json file
    with open(f'../data/{_DATA}.json', 'w') as f:
        f.write(json_data)
    df.to_csv(f"../data/{_DATA}.csv", index=False)