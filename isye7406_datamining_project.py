import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge 
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

def main():
    testing_check = True
    
    analysis = DataAnalysis(testing=testing_check)
    analysis.load_data()  
    analysis.clean_data()

    analysis.group_vars(analysis.data)

    analysis.convert_data()
    analysis.split_data(analysis.data, analysis.reg_response_field, 42)
    analysis.visualize_data(analysis.data, "multicol")

    response_fields = [analysis.reg_response_field] + analysis.class_response_fields
    x_train = analysis.train_data.drop(columns=response_fields)
    x_test = analysis.test_data.drop(columns=response_fields)
    y_train_reg = analysis.train_data[analysis.reg_response_field]
    y_train_70 = analysis.train_data["scores_over_70"]
    y_train_80 = analysis.train_data["scores_over_80"]
    y_train_90 = analysis.train_data["scores_over_90"]
    y_test_reg = analysis.test_data[analysis.reg_response_field]
    y_test_70 = analysis.test_data["scores_over_70"]
    y_test_80 = analysis.test_data["scores_over_80"]
    y_test_90 = analysis.test_data["scores_over_90"]

    x_train_80, y_train_80 = SMOTE().fit_resample(x_train, y_train_80)
    x_train_90, y_train_90 = SMOTE().fit_resample(x_train, y_train_90)

    rf_reg_params = {'n_estimators': 350, 
                    'min_samples_split': 5, 
                    'min_samples_leaf': 5, 
                    'max_depth': 85}
    
    rf_class_params = {'n_estimators': 600, 
                        'min_samples_split': 5, 
                        'min_samples_leaf': 2, 
                        'max_depth': 45}
    
    gb_class_params = {'n_estimators': 1100, 
                       'min_samples_split': 10, 
                       'max_depth': 5, 
                       'learning_rate': 0.07}
    
    pca_linear = {"n_components": 4}
    ridge_alpha = {"alpha": 15}
    lasso_alpha = {"alpha": 0.05}
    pcr_linear = {"n_features": 20}

    reg_params = [rf_reg_params, ridge_alpha, lasso_alpha, pca_linear, pcr_linear]
    reg_models = ["random_forest", "ridge", "lasso", "pca_linear", "linear_stepwise"]

    class_params = [rf_class_params, gb_class_params]
    class_models = ["random_forest", "gradient_boosting"]

    tuning_dict = {}
    tuning_dict["reg"] = analysis.tune_reg(x_train, 
        analysis.train_data[analysis.reg_response_field])

    tuning_dict["scores_over_70"] = analysis.tune_class(x_train, analysis.train_data["scores_over_70"])
    
    var_arr = analysis.get_pca_var()
    export_df(var_arr, "pca_var_ratio.csv", False)

    reg_scores = {}
    model_list = []
    for param, name in zip(reg_params, reg_models):
        reg_model = analysis.train_regression(x_train, y_train_reg, name, params=param)
        scores = analysis.eval_reg(reg_model, x_test, y_test_reg)
        reg_scores[name] = scores
        model_list.append(reg_model)

    x = 1

    mccv_reg = analysis.data.drop(columns=response_fields)
    mccv_y = analysis.data[analysis.reg_response_field]

    df_reg = analysis.run_mccv_reg(mccv_reg, mccv_y)
    # df_class = analysis.run_mccv_class(total_70, y_70, "scores_over_70")
    # export_df(df_class, "scores_over_70_mccv.csv")
    # df_class = analysis.run_mccv_class(total_80, y_80, "scores_over_80")
    # export_df(df_class, "scores_over_80_mccv.csv")
    # df_class = analysis.run_mccv_class(total_90, y_90, "scores_over_90")
    # export_df(df_class, "scores_over_90_mccv.csv")

    class_scores = {}
    for param, name in zip(class_params, class_models):
        class_model = analysis._train_classifier(x_train_90, y_train_90, name, params=param)
        score = analysis.eval_class(class_model, x_test, y_test_90)
        class_scores[name + "90"] = score
        model_list.append(class_model)
    
    for param, name in zip(class_params, class_models):
        class_model = analysis._train_classifier(x_train_80, y_train_80, name, params=param)
        score = analysis.eval_class(class_model, x_test, y_test_80)
        class_scores[name + "80"] = score
        model_list.append(class_model)

    for param, name in zip(class_params, class_models):
        class_model = analysis._train_classifier(x_train, y_train_70, name, params=param)
        score = analysis.eval_class(class_model, x_test, y_test_70)
        class_scores[name + "70"] = score
        model_list.append(class_model)

    feat_importances = model_list[0].feature_importances_
    feat_importances = np.column_stack((
        feat_importances, model_list[5].feature_importances_
    ))
    for i in range(6, 11):
        feat_importances = np.concatenate(
            (feat_importances, model_list[i].feature_importances_.reshape((-1,1))),
            axis=1
        )
    cols = ["rf_reg", "rf_90", "gb_90", "rf_80", "gb_80", "rf_70", "gb_70", "ridge", "lasso", "rfe"]
    feat_importances = np.concatenate((feat_importances, model_list[1].coef_.reshape((-1,1))), axis=1)
    feat_importances = np.concatenate((feat_importances, model_list[2].coef_.reshape((-1,1))), axis=1)
    feat_importances = np.concatenate((feat_importances, model_list[4].ranking_.reshape((-1,1))), axis=1)

    feat_import_df = pd.DataFrame(feat_importances, columns=cols)
    feat_import_df["feature_names"] = model_list[0].feature_names_in_
    export_df(feat_import_df, "feature_importances.csv")
    


def check_na(analysis_obj):
    # Checks to see which columns have missing data
    col_list = list(analysis_obj.data.columns)
    na_list = list(analysis_obj.data.isnull().values.any(axis=0))
    col_count = 0

    for col, has_na in zip(col_list, na_list):
        if has_na:
            print(f"{col} has missing data")
            col_count += 1

    if col_count == 0:
        print("Data does not have missing values")

def export_df(df, file_name, is_df=True):
    if not is_df:
        df = pd.DataFrame(df)
    
    dir_name = os.path.dirname(__file__)
    file_path = f"{dir_name}\\results\\{file_name}"
    
    if is_df:
        df.to_csv(file_path, index=None)
    else:
        df.to_csv(file_path, header=None, index=None)
    
    print(f"{file_name} is exported")

def test_cleaning(analysis_obj):
    # Function to check the cleaning steps
    print(analysis_obj.data.columns)
    check_na(analysis_obj)

class DataAnalysis(object):
    def __init__(self, testing=True) -> None:
        self.data = None
        self.reg_response_field = "exam_score"
        self.class_response_fields = ["scores_over_70", "scores_over_80", "scores_over_90"]
        self.dir_name = os.path.dirname(__file__)
        self.dir_graphics = f"{self.dir_name}\\graphics\\"
        self.dir_results = f"{self.dir_name}\\results\\"
        self.reg_models = dict()
        self.class_models = dict()
        self.cluster_models = dict()
        self.reg_scores = dict()
        self.class_scores = dict()
        self.cluster_scores = dict()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.testing = testing
        self.categorical_cols = ["parental_involvement", "access_to_resources", 
                                 "extracurricular_activities", "motivation_level", 
                                 "internet_access", "family_income", "teacher_quality",
                                 "school_type", "peer_influence", 
                                 "learning_disabilities", "parental_education_level",
                                 "distance_from_home", "gender"]
        self.categorical_log = "Categorical variable conversions\n********"
        self.train_data = None
        self.test_data = None
    
    def load_data(self):
        file_name = "StudentPerformanceFactors.csv"
        file_path = f"{self.dir_name}\\data\\{file_name}"
        self.data = pd.read_csv(file_path)
        
    def clean_data(self):
        df = self.data.copy()
        df = self._rename_cols(df)
        df = self._impute_data(df)

        self.data = df

    def convert_data(self):

        df = self.data.copy()
        df = self._transform_categoricals(df, self.categorical_cols)
        df = self._classify_response(df, self.reg_response_field)
        col_list = list(df.columns)
        response_col = df[self.reg_response_field]
        df = MinMaxScaler().fit_transform(df)

        df = pd.DataFrame(df, columns=col_list)
        df[self.reg_response_field] = response_col
        self.data = df
    
    def group_vars(self, df):
        col_list = list(df.columns)

        group_df = df.groupby(by=col_list[0]).count().reset_index()
        df_1 = group_df[[col_list[0], "attendance"]].copy()
        df_1["factor"] = col_list[0]
        df_1 = df_1.rename(columns={col_list[0]: "level", "attendance": "count"})

        for col in col_list[1:]:
            group_df = df.groupby(by=col).count().reset_index()
            temp_df = group_df[[col_list[0], col]].copy()
            temp_df["factor"] = col
            temp_df = temp_df.rename(columns={col: "level", col_list[0]: "count"})
            df_1 = pd.concat((df_1, temp_df), axis=0)
        
        export_df(df_1, "levels_check.csv", is_df=True)

    def _rename_cols(self, df):
        # Renames columns - now from upper case to lower case
        old_cols = list(df.columns)
        new_cols = [x.lower() for x in old_cols]
        col_mapper = dict(zip(old_cols, new_cols))
        df = df.rename(columns=col_mapper)
        return df

    def split_data(self, df, response, rand_number):
        train_data, test_data = train_test_split(
            df, test_size=0.2, random_state=rand_number
        )

        # train_data = train_data.sort_index()
        # test_data = test_data.sort_index()

        self.train_data = train_data.sort_index()
        self.test_data = test_data.sort_index()

        # self.X_train = train_data.drop(columns=response)
        # self.X_test = test_data.drop(columns=response)
        # self.y_train = train_data[response]
        # self.y_test = test_data[response]

    def visualize_data(self, df, plot_type="histogram"):
        viz_types = ["histogram", "multicol", "var_ratio"]

        if plot_type not in viz_types:
            print("Plot type does not match a graph type")
        
        elif plot_type == viz_types[0]:
            self._plot_subplots(df)

        elif plot_type == viz_types[1]:
            corr_data = df.corr()
            corr_data = corr_data.abs()
            sns.heatmap(corr_data, cmap="viridis")
            file_name = "corr_heatmap.png"

        elif plot_type == viz_types[2]:
            #TODO: Add PCA variance ratio plot
            pass        
        
        if plot_type != viz_types[0]:
            file_path = f"{self.dir_name}\\graphics\\{file_name}"
            plt.savefig(file_path)

    def _plot_subplots(self, df):
        features = list(df.columns)
        fig, axes = self._gen_subplots(len(features), row_wise=True)

        for feature, ax in zip(features, axes):
            sns.histplot(df[feature], ax=ax)

        plt.tight_layout(pad=0.5)

        file_name = "var_distro.png"
        plt.savefig(f"{self.dir_graphics}{file_name}")

    def _gen_subplots(self, num_plots, row_wise=False):
        #TODO: Finish this func
        n_rows, n_cols = self._subplot_dimensions(num_plots)
        fig, axes = plt.subplots(n_rows, n_cols, sharex=False, sharey=True)

        if not isinstance(axes, np.ndarray):
            return fig, [axes]
        else:
            axes = axes.flatten(order=('C' if row_wise else 'F'))
            
            for idx, ax in enumerate(axes[num_plots:]):
                fig.delaxes(ax)

                idx_ticks = idx + num_plots - n_cols if row_wise else idx + num_plots - 1
                for tk in axes[idx_ticks].get_xticklabels():
                    tk.set_visible(True)

            axes = axes[:num_plots]
            return fig, axes
        
    def _subplot_dimensions(self, num_plots):
        if num_plots <= 4:
            return num_plots, 1
        else:
            return self._ceil_div(num_plots, 4), 4
        
    def _ceil_div(self, a, b):
        return -(a // -b)

    def _impute_data(self, df):
        # Imputes data with the mode
        cols = list(df.columns)
        null_list = list(df.isna().values.any(axis=0))

        for col, has_na in zip(cols, null_list):
            if has_na:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df

    def _transform_categoricals(self, df, categorical_cols):
        # Goes thru list of categorical columns and converts them to onehot cols
        # Then drops the columns from the data
        for field in categorical_cols:
            unique_list = df[field].unique()
            df = self._create_onehots(df, field, unique_list)
        
        df = df.drop(columns=self.categorical_cols)
        return df

    def _classify_response(self, df, response_col):
        response_levels = [70, 80, 90]

        for level in response_levels:
            mask = df[response_col] >= level
            new_field = f"scores_over_{level}"
            df[new_field] = 0
            df.loc[mask, new_field] = 1
            self.class_response_fields.append(new_field)

        return df

    def _create_onehots(self, df, field, uniques):
        # Converts columns to one-hot columns
        self.categorical_log += f"\n{field} - "
        for i, value in enumerate(uniques):
            mask = df[field] == value
            new_field = f"{field}_{i}"
            df[new_field] = 0
            df.loc[mask, new_field] = 1
            self.categorical_log += f"{i}: {value}"
        
        self.categorical_log = self.categorical_log[:-2]           
        return df
    
    def train_regression(self, X_train, y_train, model_name, params=None):
        model_list = ["random_forest", "ridge", "lasso", "pca_linear", "linear_stepwise",
                      "linear_reg"]

        if params is not None:
            param_dict = params
        elif model_name == model_list[0]:
            param_dict = {"n_estimators": 100}
        elif model_name == model_list[1]:
            param_dict = {"alpha": 0.5}
        elif model_name == model_list[2]:
            param_dict = {"alpha": 0.1}
        elif model_name == model_list[3]:
            param_dict = {"n_components": 4}
        elif model_name == model_list[4]:
            param_dict = {"n_features": None}
        else:
            param_dict = None

        if model_name not in model_list:
            print(f"{model_name} not an available model")
        elif model_name == model_list[0]:
            return RandomForestRegressor(**param_dict).fit(X_train, y_train)
        elif model_name == model_list[1]:
            return Ridge(alpha=param_dict["alpha"]).fit(X_train, y_train)
        elif model_name == model_list[2]:
            return Lasso(alpha=param_dict["alpha"]).fit(X_train, y_train)
        elif model_name == model_list[3]:
            pca = PCA(n_components=param_dict["n_components"]).fit(X_train)
            x_transform = pca.transform(X_train)
            return LinearRegression().fit(x_transform, y_train), pca
        elif model_name == model_list[4]:
            return RFE(LinearRegression(), n_features_to_select=param_dict["n_features"]).fit(X_train, y_train)
        elif model_name == model_list[5]:
            return LinearRegression().fit(X_train, y_train)
    
    def _train_classifier(self, X_train, y_train, model_name, params=None):
        model_list = ["random_forest", "gradient_boosting"]

        if params is not None:
            param_dict = params
        elif model_name == model_list[0]:
            param_dict = {"n_estimators": 100}
        elif model_name == model_list[1]:
            param_dict = {"alpha": 0.5}
        else:
            param_dict = None

        if model_name not in model_list:
            print(f"{model_name} not an available model")
        elif model_name == model_list[0]:
            return RandomForestClassifier(**param_dict).fit(X_train, y_train)
        elif model_name == model_list[1]:
            return GradientBoostingClassifier(**param_dict).fit(X_train, y_train)
        
    def get_pca_var(self, X_train):
        pca = PCA().fit(X_train)
        return pca.explained_variance_ratio_
        
    def train_linreg(x_data, y_data):
        return LinearRegression().fit(x_data, y_data)  

    def tune_reg(self, x_data, y_data):       
        rand_state = 42
        n_estimators = [int(x) for x in np.linspace(100, 1100, 6)]
        max_depth = [int(x) for x in np.linspace(5, 105, num=6)]
        min_leaf = [2, 5, 10]
        min_split = [5, 10, 15]
        alphas_ridge = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15]
        alphas_lasso = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

        # TODO: Implement CV for RFE
        
        rf_dict = {"n_estimators": n_estimators,
                   "max_depth": max_depth,
                   "min_samples_leaf": min_leaf,
                   "min_samples_split": min_split}

        rf_cv = RandomizedSearchCV(
            estimator=RandomForestRegressor(), param_distributions=rf_dict,
            n_iter=100, verbose=2, random_state=rand_state, n_jobs=-1
        )

        ridge_cv = RidgeCV(alphas=alphas_ridge, cv=100)
        lasso_cv = LassoCV(alphas=alphas_lasso, cv=100)

        cv_list = [rf_cv, ridge_cv, lasso_cv]
        best_params = {}

        for cv in cv_list:
            cv.fit(x_data, y_data)
        
        best_params["rf"] = cv_list[0].best_params_
        best_params["ri"] = cv_list[1].alpha_
        best_params["la"] = cv_list[2].alpha_

        return best_params    

    def tune_class(self, x_data, y_data):
        rand_state = 42
        n_estimators = [int(x) for x in np.linspace(100, 1100, 5)]
        max_depth = [int(x) for x in np.linspace(5, 105, num=6)]
        min_leaf = [2, 5, 10]
        min_split = [5, 10, 15]
        learning_rate = [x for x in np.linspace(0.01, 0.11, num=6)]

        rf_dict = {"n_estimators": n_estimators,
                   "max_depth": max_depth,
                   "min_samples_leaf": min_leaf,
                   "min_samples_split": min_split}
        
        gb_dict = {"n_estimators": n_estimators,
                   "max_depth": max_depth,
                   "min_samples_split": min_split,
                   "learning_rate": learning_rate}
        
        rf_cv = RandomizedSearchCV(
            estimator=RandomForestClassifier(), param_distributions=rf_dict,
            n_iter=100, verbose=2, random_state=rand_state, n_jobs=-1
        )

        gb_cv = RandomizedSearchCV(
            estimator=GradientBoostingClassifier(), param_distributions=gb_dict,
            n_iter=2, verbose=2, random_state=rand_state, n_jobs=-1
        )
        
        cv_list = [rf_cv, gb_cv]

        for cv in cv_list:
            cv.fit(x_data, y_data)
        
        best_params = {}
        best_params["rf"] = cv_list[0].best_params_
        best_params["lg"] = cv_list[1].best_params_

        return best_params

    def eval_reg(self, model, x_test, y_test):
        if isinstance(model, tuple):
            new_x = model[1].transform(x_test)
            y_pred = model[0].predict(new_x)
        else:
            y_pred = model.predict(x_test)
        
        scores_list = []
        for i in range(1, 6):
            dif = np.abs(y_pred - y_test)
            counts = np.count_nonzero(dif <= i)
            score = round(counts / len(dif), 4)
            scores_list.append(score)

        return scores_list

    def eval_class(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        return score

    def _calc_mse(self, y_pred, y_test):
        mse = (y_test - y_pred) ** 2
        mse = np.sum(mse) / len(y_pred)
        return mse

    def run_mccv_reg(self, X_data, y_data):
        reg_models = ["random_forest", "ridge", "lasso", "pca_linear", "linear_stepwise"]
        n_iters = 100

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.1, random_state=0
        )

        models = self._reg_for_mccv(X_train, y_train)
        scores = self._score_reg_mccv(models, X_test, y_test)
        row_dict = dict(zip(reg_models, scores))
        df_1 = pd.DataFrame(row_dict)
        df_1["score_diff"] = [1, 2, 3, 4, 5]

        print("Iteration: 0")

        for i in range(1, n_iters):
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.1, random_state=i
            )

            models = self._reg_for_mccv(X_train, y_train)
            scores = self._score_reg_mccv(models, X_test, y_test)
            row_dict = dict(zip(reg_models, scores))
            temp_df = pd.DataFrame(row_dict)
            temp_df["score_diff"] = [1, 2, 3, 4, 5]
            df_1 = pd.concat((df_1, temp_df))
            print(f"Iteration: {i}")

        return df_1

    def _reg_for_mccv(self, x_data, y_data):
        rf_reg_params = {'n_estimators': 350, 
                    'min_samples_split': 5, 
                    'min_samples_leaf': 5, 
                    'max_depth': 85}
        
        pca_linear = {"n_components": 4}
        ridge_alpha = {"alpha": 15} 
        lasso_alpha = {"alpha": 0.05}
        pcr_linear = {"n_features": 20}

        reg_params = [rf_reg_params, ridge_alpha, lasso_alpha, pca_linear, pcr_linear]
        reg_models = ["random_forest", "ridge", "lasso", "pca_linear", "linear_stepwise"]
        models_list = []

        for model, param_dict in zip(reg_models, reg_params):
            models_list.append(self.train_regression(x_data, y_data, model, param_dict))

        return models_list
    
    def _score_reg_mccv(self, models, x_test, y_test):
        score_list = []
        for model in models:
            score = self.eval_reg(model, x_test, y_test)
            score_list.append(score)

        return score_list
    
    def run_mccv_class(self, X_data, y_data, response_field):
        class_models = ["random_forest", "gradient_boosting"]
        n_iters = 100

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.1, random_state=0
        )

        models = self._class_for_mccv(X_train, y_train)
        scores = self._score_class_mccv(models, X_test, y_test)
        row_dict = dict(zip(class_models, scores))
        df_1 = pd.DataFrame(row_dict)
        df_1["response_field"] = response_field

        print("Iteration: 0")

        for i in range(1, n_iters):
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.1, random_state=i
            )

            models = self._class_for_mccv(X_train, y_train)
            scores = self._score_class_mccv(models, X_test, y_test)
            row_dict = dict(zip(class_models, scores))
            temp_df = pd.DataFrame(row_dict)
            temp_df["response_field"] = response_field
            df_1 = pd.concat((df_1, temp_df))
            print(f"Iteration: {i}")

        return df_1
    
    def _class_for_mccv(self, x_data, y_data): 
        rf_class_params = {'n_estimators': 600, 
                           'min_samples_split': 5, 
                           'min_samples_leaf': 2, 
                           'max_depth': 45}
        
        gb_class_params = {'n_estimators': 1100, 
                           'min_samples_split': 10, 
                           'max_depth': 5, 
                           'learning_rate': 0.07}

        class_params = [rf_class_params, gb_class_params]
        class_models = ["random_forest", "gradient_boosting"]
        models_list = []

        for model, param_dict in zip(class_models, class_params):
            models_list.append(self._train_classifier(x_data, y_data, model, param_dict))

        return models_list
    
    def _score_class_mccv(self, models, x_test, y_test):
        score_list = []
        for model in models:
            y_pred = model.predict(x_test)
            score = accuracy_score(y_test, y_pred)
            score_list.append([score])

        return score_list

if __name__ == "__main__":
    main()