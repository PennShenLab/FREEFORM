import os
import utils
import warnings
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression

def evaluate_feature_engineering(data_name, shots, seeds, method_name=None):
    results = {}
    
    for shot in shots:
        print(f"Evaluating {method_name} for {shot}-shot")
        results[shot] = {'accuracy': [], 'auc': [], 'f1': []}
        
        for seed in seeds:
            utils.set_seed(seed)
            
            # Assume get_dataset retrieves your data according to the shot and seed
            df, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = utils.get_dataset(data_name, shot, seed)
            
            # Turn all labels numeric
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            
            # Set up parameter grids for grid search
            if method_name == "lr":
                model = LogisticRegression(max_iter=1000)
                param_grid = {
                    'C': [1e-4,1e-3,1e-2, 1e-1, 1, 10,100],
                    'penalty': ['l1','l2'],
                    'solver': ['saga']
                }
            elif method_name == "rf":
                model = RandomForestClassifier(random_state=seed)
                param_grid = {
                    'n_estimators': [100, 250],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif method_name == "xgboost":
                model = XGBClassifier(eval_metric='mlogloss')
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            elif method_name == "tabpfn":
                model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
                param_grid = None  # TabPFN does not use traditional hyperparameters
            else:
                raise ValueError("Unknown method")

            # Perform grid search if applicable
            if param_grid is not None:
                grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=2 if shot == 10 or shot == 4 else 4, n_jobs=4)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                # best_model=model
            else:
                best_model = model
                
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            best_model.fit(X_train, y_train)
            
            if method_name == "tabpfn":
                y_pred, y_pred_probs = best_model.predict(X_test, return_winning_probability=True)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                if len(np.unique(y_train)) == 2:
                    auc = roc_auc_score(y_test, y_pred_probs)
                else:
                    # Calculate the OvR ROC AUC
                    classes = np.unique(y_train)
                    auc = calculate_ovr_roc_auc(y_test, y_pred, y_pred_probs, classes)
                    
            else:
                y_pred = best_model.predict(X_test)
                y_pred_probs = best_model.predict_proba(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
        
                if len(np.unique(y_train)) == 2:  # Binary classification
                    auc = roc_auc_score(y_test, y_pred_probs[:, 1])
                else:  # Multi-class classification
                    auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')

            results[shot]['accuracy'].append(accuracy)
            results[shot]['auc'].append(auc)
            results[shot]['f1'].append(f1)
    
    return results


def calculate_ovr_roc_auc(y_true, y_pred, y_pred_probs, classes):
    """
    Manually calculate the multiclass ROC AUC using the One-vs-Rest approach.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_pred_probs: Predicted probabilities for the predicted class (1D array)
    :param classes: List of class labels
    :return: Average ROC AUC score across all classes
    """
    ovr_roc_auc = []

    for cls in classes:
        # Create binary labels for the current class
        binary_y_true = np.where(y_true == cls, 1, 0)
        
        # For TabPFN, we only have the probability for the predicted class.
        # Use the probability directly if the predicted class is `cls`,
        # and `1 - probability` otherwise.
        binary_y_score = np.where(y_pred == cls, y_pred_probs, 1 - y_pred_probs)
        
        # Calculate the ROC AUC for the current class
        try:
            auc = roc_auc_score(binary_y_true, binary_y_score)
            ovr_roc_auc.append(auc)
        except ValueError:
            # This can happen if a class has only one label in y_true
            print(f"Skipping ROC AUC calculation for class {cls} due to lack of positive/negative samples.")
            continue
    
    # Calculate the average ROC AUC across all classes
    average_ovr_roc_auc = np.mean(ovr_roc_auc) if ovr_roc_auc else None
    
    return average_ovr_roc_auc

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


            
def print_results(results, shots, method_name):
    table = []
    
    for shot in shots:
        avg_accuracy = np.mean(results[shot]['accuracy'])
        avg_auc = np.mean(results[shot]['auc'])
        avg_f1 = np.mean(results[shot]['f1'])
        std_accuracy = np.std(results[shot]['accuracy']) 
        std_auc = np.std(results[shot]['auc']) 
        std_f1 = np.std(results[shot]['f1']) 
        
        table.append([
            shot, 
            f"{avg_accuracy:.4f}", f"{std_accuracy:.3f}", 
            f"{avg_auc:.4f}", f"{std_auc:.3f}", 
            f"{avg_f1:.4f}", f"{std_f1:.3f}"
        ])
    
    headers = ["Shot", "Avg Accuracy", "Std", "Avg AUC", "Std", "Avg F1", "Std"]
    print(f"Results for {method_name}")
    print(tabulate(table, headers=headers, tablefmt="pretty"))


def plot_results(data_name, ml_name, results_dict, shots, save_path='./plots', save=False,type = "feature_selection"):
    
    
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
        plt.plot(x_positions, avg_accuracies, marker='o', label=method)
    
    if data_name == "Ancestry" and type == "feature_engineering":
        plt.ylim(0.8, 1.0)
        
    # Set the custom ticks and labels
    plt.xticks(ticks=x_positions, labels=shots)
    plt.xlabel('Number of Shots')
    plt.ylabel('AUC (%)')
    
    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.2))
    #plt.legend()
    plt.grid()
    # Save the plot as a PDF
    if save:
        file_name = f'{data_name}_{ml_name}_{type}.pdf'
        plt.savefig(os.path.join(save_path, file_name), format='pdf', bbox_inches='tight')
        
    plt.show()
    
def plot_results_multiple(data_name, ml_name, results_dict_list, shots_list, titles_list=None,
                          save_path='./plots', plot_type='feature_selection', save=False):
    # Number of plots
    n_plots = len(results_dict_list)
    
    # Determine grid size
    if n_plots <= 3:
        nrows = 1
        ncols = n_plots
    else:
        nrows = 2
        ncols = int(np.ceil(n_plots / nrows))

    # Create subplots with squeeze=False
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*5), squeeze=False)
    
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
    
    # Iterate over each plot
    for idx, (results_dict, shots, ax) in enumerate(zip(results_dict_list, shots_list, axes_flat)):
        x_positions = range(len(shots))
        
        for method, results in results_dict.items():
            avg_accuracies = [np.mean(results[shot]['auc']) for shot in shots]
            wrapped_method = method.replace('+ ', '+\n')
            
            # Determine line style
            linestyle = '-' if 'FREEFORM' in method else ':'
            
            # Plot data
            line, = ax.plot(x_positions, avg_accuracies, marker='o', linestyle=linestyle, label=wrapped_method)
            
            # Collect legend entries
            legend_entries.append((wrapped_method, line))
        
        # Set x-ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(shots)
        
        # Set title
        if titles_list and idx < len(titles_list):
            ax.set_title(titles_list[idx])
        else:
            ax.set_title(f'Plot {idx+1}')

        ax.set_xlabel('Number of Shots')
        
        # Set y-axis limits for the first plot
        if idx == 0:
            ax.set_ylim(0.8, 0.985)
        
        # Show y-tick labels for every subplot, but only show y-axis title for the leftmost subplots
        if idx % ncols == 0:
            ax.set_ylabel('AUROC')  # Show y-axis title for leftmost plots
        else:
            ax.set_ylabel('')  # Hide y-axis title for non-leftmost subplots
        ax.grid(True)
    
    # Hide unused subplots
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)
    
    # Remove duplicate legend entries
    unique_legend_entries = {}
    for label, handle in legend_entries:
        if label not in unique_legend_entries:
            unique_legend_entries[label] = handle

    # Separate legend entries
    non_freeform_entries = []
    freeform_entries = []
    for label, handle in unique_legend_entries.items():
        if 'FREEFORM' in label:
            freeform_entries.append((label, handle))
        else:
            non_freeform_entries.append((label, handle))

    # Combine entries
    ordered_legend_entries = non_freeform_entries + freeform_entries

    # Unpack labels and handles
    legend_labels, legend_handles = zip(*ordered_legend_entries)

    # Adjust layout to make room for the legend at the bottom
    plt.subplots_adjust(bottom=0.25)  # Create more space at the bottom for the legend

    # Calculate number of columns for legend
    ncol = int(np.ceil(len(legend_labels) / 2))
    
    # Add legend below the plots
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, -0.1))
    
    # Save the plot
    file_name = f'{data_name}_{ml_name}_{plot_type}.pdf'
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, file_name), format='pdf', bbox_inches='tight')
    
    plt.show()
     
def plot_results_with_error(data_name, results_dict, shots):
    for method, results in results_dict.items():
        avg_aucs = [np.mean(results[shot]['auc']) for shot in shots]
        std_aucs = [np.std(results[shot]['auc']) for shot in shots]

        plt.plot(shots, avg_aucs, marker='o', label=method)
        plt.fill_between(shots, 
                         np.array(avg_aucs) - np.array(std_aucs), 
                         np.array(avg_aucs) + np.array(std_aucs), 
                         alpha=0.2)

    plt.xlabel('Shot Size')
    plt.ylabel('Average AUC')
    plt.title(f'Shot Feature Engineering Method Comparison for {data_name}')
    
    if data_name == "Ancestry":
        plt.ylim(0.8, 1.0)
    
    plt.legend()
    plt.show()