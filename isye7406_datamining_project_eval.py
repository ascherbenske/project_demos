import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os
from itertools import combinations

def main():
    df = load_data("mccv_reg.csv")
    eval_reg(df)

    class_dict = {"stat": [],
                  "pvalue": []}
    
    thresholds = [70, 80, 90]
    class_dict["thresholds"] = thresholds

    for val in thresholds:
        df = load_data(f"scores_over_{val}_mccv.csv")
        mean_val = df.drop(columns="response_field").mean()
        print(f"{val}: {mean_val}")
        wil_stat, wil_p = eval_class(df)
        class_dict["stat"].append(wil_stat)
        class_dict["pvalue"].append(wil_p)

    df = pd.DataFrame(class_dict)
    export_df(df, "class_wilc.csv")

def load_data(file_name):
    current_dir = os.path.dirname(__file__)
    file_path = f"{current_dir}\\results\\{file_name}"
    return pd.read_csv(file_path)

def eval_class(df):
    cols = ["random_forest", "gradient_boosting"]
    diff = df[cols[0]] - df[cols[1]]
    if abs(sum(diff)) > 0:
        result = wilcoxon(df[cols[0]], df[cols[1]])
        return result.statistic, result.pvalue
    else:
        return -1, 1

def eval_reg(df):
    score_diffs = list(range(1, 6))
    means = df.groupby(by="score_diff").mean().reset_index()
    export_df(means, "reg_means_wilcoxon")

    models = ["random_forest", "ridge", "lasso", "pca_linear", "linear_stepwise"]
    model_combos = list(combinations(models, r=2))

    for diff in score_diffs:
        temp_df = df.query("score_diff == @diff")

        model_stats = {"model_1": [],
                       "model_2": [],
                       "stat": [],
                       "pvalue": []}

        for combo in model_combos:
            col_diff = temp_df[combo[0]] - temp_df[combo[1]]
            model_stats["model_1"].append(combo[0])
            model_stats["model_2"].append(combo[1])

            if abs(sum(col_diff)) > 0:            
                result = wilcoxon(temp_df[combo[0]], temp_df[combo[1]])
                model_stats["stat"].append(result.statistic)
                model_stats["pvalue"].append(result.pvalue)
            else:
                model_stats["stat"].append(-1)
                model_stats["pvalue"].append(1)

        df_wil = pd.DataFrame(model_stats)
        export_df(df_wil, f"reg_wilcoxon_diff{diff}")
    
    print("Reg Wilcoxon Exported")

def export_df(df, file_name):
    dir_name = os.path.dirname(__file__)
    file_path = f"{dir_name}\\results\\{file_name}.csv"
    df.to_csv(file_path, index=None)
    print(f"{file_name} exported")

if __name__ == "__main__":
    main()