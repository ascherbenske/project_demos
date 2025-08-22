import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb


def main():
    testing = True
    
    analysis = DataAnalysis()
    analysis.load_data('synth_data_20250310.csv')
    analysis.prep_data()
    analysis.split_data()

    X_data = analysis.data.drop(columns='delay').copy()
    y_data = analysis.data.delay.copy()
    # analysis.tune_models(X_data, y_data)

    analysis.run_mccv(X_data, y_data)

    model_list = ['xgboost', 'gradient boosting', 'random forest']
    # model_list = ['gradient boosting', 'logistic regression', 'random forest']
    
    data_dict = {"model":[], "response":[], "percentage":[], "pred_response":[], "real_response":[]}
    df_list = []
    for model_name in model_list:
        print(f'{model_name} start analysis')
        model = analysis._train_model(analysis.X_train, analysis.y_train, model_name)
        probabilities = model.predict_proba(analysis.X_test)
        pred_response = model.predict(analysis.X_test)
        n_points = probabilities.shape[0]
        classes = model.classes_
        
        class_cols = [f'class_{x}' for x in classes]
        df_temp = pd.DataFrame(probabilities, columns=class_cols)
        df_temp['real_response'] = analysis.y_test
        df_temp['pred_response'] = pred_response
        df_temp['model'] = model_name
        df_list.append(df_temp)

        for i, response in enumerate(classes):
            data_dict['model'].extend([model_name] * n_points)
            data_dict['response'].extend([response] * n_points)
            data_dict['percentage'].extend(list(probabilities[:, i]))
            data_dict['pred_response'].extend(list(pred_response))
            data_dict['real_response'].extend(list(analysis.y_test))
    
    df = pd.DataFrame(data_dict)
    df['floor'] = np.floor(df.percentage * 10) / 10
    df['ceil'] = np.ceil(df.percentage * 10) / 10
    
    export_df(df, 'probs_test')

    df_to_export = df_list[0]
    for data in df_list[1:]:
        df_to_export = pd.concat((df_to_export, data))
    export_df(df_to_export, 'probs_combined')


def export_df(df, file_name, loc=0):
    dir_name = os.path.dirname(__file__)

    if loc == 0:
        folder = 'results'

    file_path = f'{dir_name}\\{folder}\\{file_name}.csv'
    df.to_csv(file_path, index=None)
    print(f'{file_name} is exported to {file_path}')

def export_array(np_array, file_name, loc=0):
    dir_name = os.path.dirname(__file__)
    df = pd.DataFrame(np_array)

    if loc == 0:
        folder = 'results'
    
    file_path = f'{dir_name}\\{folder}\\{file_name}.csv'
    df.to_csv(file_path, index=None)
    print(f'{file_name} is exported to {file_path}')

class DataAnalysis(object):
    def __init__(self):
        self.raw_data = None
        self.data = None
        self.dir_name = os.path.dirname(__file__)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.response_field = 'delay'
        self.models = {}
        self.scores = {}
    
    def load_data(self, file_name):
        file_path = f'{self.dir_name}\\{file_name}'
        self.raw_data = pd.read_csv(file_path)
    
    def clean_data(self):
        df = self.raw_data.copy()
        
        drop_list = ['ID', 
                     'weight_in_progress', 
                     'weight_not_functional', 
                     'weight_trial_ready',
                     'cd_id',
                     'project']
        
        col_list = list(df.columns)
        drop_list = [x for x in drop_list if x in col_list]
        
        self.data = df.drop(columns=drop_list)
    
    def prep_data(self):
        # self.clean_data()
        self.clean_data()
        self.transform_data()
        self.scale_data()
    
    def scale_data(self):
        df = self.data.copy()
        scale_list = ['week_number',
                      'duration',
                      'start_week',
                      'weeks_until_req_end',
                      'weeks_until_proj_end']
        
        for feature in scale_list:
            new_feature = f'{feature}_clean'
            df[new_feature] = (MinMaxScaler()
                                   .fit_transform(
                                       np.reshape(df[feature], (-1,1))))

        self.data = df

    def transform_data(self):
        project_end = 125
        df = self.data.copy()
        df['weeks_until_req_end'] = df['duration'] + df['start_week'] - df['week_number']
        df['weeks_until_proj_end'] = project_end - df['week_number']

        self.data = df

    def split_data(self, test_pct=0.1, fix_rand=True):
        if fix_rand:
            train_data, test_data = train_test_split(
                self.data, test_size=test_pct, random_state=42
            )
        else:
            train_data, test_data = train_test_split(
                self.data, test_size=test_pct
            )
        
        train_data = train_data.sort_index().reset_index()
        test_data = test_data.sort_index().reset_index()

        self.X_train = train_data.drop(columns=self.response_field)
        self.y_train = train_data[self.response_field]
        self.X_test = test_data.drop(columns=self.response_field)
        self.y_test = test_data[self.response_field]

    def _train_model(self, X_train, y_train, model_name='random forest', params=None):
        if params is not None:
            param_dict = params
        elif model_name == 'xgboost':
            param_dict = {'n_estimators': 200,
                          'max_depth': 65,
                          'grow_policy': 'lossguide',
                          'learning_rate': 0.09}
        elif model_name == 'random forest':
            param_dict = {'n_estimators': 500,
                          'min_samples_leaf': 5,
                          'min_samples_split': 15,
                          'min_impurity_decrease': 0.05,
                          'max_depth': 85,
                          'criterion': 'entropy'}       
        elif model_name == 'gradient boosting':
            param_dict = {'n_estimators': 400,
                          'min_samples_leaf': 5,
                          'min_samples_split': 5,
                          'min_impurity_decrease': 0.0,
                          'max_depth': 5}
        elif model_name == 'logistic regression':
            param_dict = {'penalty': 'elasticnet',
                          'solver': 'saga',
                          'l1_ratio': 0.5,
                          'C': 1.0}       
        
        if model_name == 'random forest':
            return RandomForestClassifier(**param_dict).fit(X_train, y_train)
        elif model_name == 'gradient boosting':
            return GradientBoostingClassifier(**param_dict).fit(X_train, y_train)
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(**param_dict).fit(X_train, y_train)
        elif model_name == 'logistic regression':
            return LogisticRegression(**param_dict).fit(X_train, y_train)
        
    def tune_models(self, x_data, y_data):
        rand_state = 42

        n_estimators = [int(x) for x in np.linspace(100, 500, 5)]
        max_depth = [int(x) for x in np.linspace(5, 105, num=6)]
        learning_rate = [x for x in np.linspace(0.01, 0.1, num=10)]
        min_impurity = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        min_leaf = [2, 5, 10]
        min_split = [5, 10, 15]
        criterion = ['gini', 'entropy', 'log_loss']
        C_val = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5, 10]
        solver = 'saga'
        l1_ratio = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        grow_policy = ['depthwise', 'lossguide']

        rf_dict = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_leaf': min_leaf,
                   'min_samples_split': min_split,
                   'min_impurity_decrease': min_impurity,
                   'criterion': criterion}
        
        gb_dict = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_split,
                   'learning_rate': learning_rate,
                   'min_impurity_decrease': min_impurity}
        
        xgb_dict = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'learning_rate': learning_rate,
                   'grow_policy': grow_policy}
        
        log_dict = {'C': C_val,
                    'solver': [solver],
                    'l1_ratio': l1_ratio,
                    'max_iter': [200]}

        iter_num = 500
        verbosity=2
        
        rf_cv = RandomizedSearchCV(
            estimator=RandomForestClassifier(), param_distributions=rf_dict,
            n_iter=iter_num, verbose=verbosity, random_state=rand_state, n_jobs=-1
        )
        
        gb_cv = RandomizedSearchCV(
            estimator=GradientBoostingClassifier(), param_distributions=gb_dict,
            n_iter=iter_num, verbose=verbosity, random_state=rand_state, n_jobs=-1
        )

        xg_cv = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(), param_distributions=xgb_dict,
             n_iter=iter_num, verbose=verbosity, random_state=rand_state, n_jobs=-1
        )

        lo_cv = RandomizedSearchCV(
            estimator=LogisticRegression(), param_distributions=log_dict,
            n_iter=iter_num, verbose=verbosity, random_state=rand_state, n_jobs=-1
        )

        cv_list = [rf_cv, gb_cv, xg_cv]
        # cv_list = [lo_cv]

        for cv in cv_list:
            cv.fit(x_data, y_data)

        best_params = {}
        best_params["rf"] = cv_list[0].best_params_
        best_params["gb"] = cv_list[1].best_params_
        best_params["xg"] = cv_list[2].best_params_

    def _score_models(self, for_mccv=True):
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            score = accuracy_score(self.y_test, y_pred)
            if for_mccv:
                self.scores[name] = [score]
            else:
                self.scores[name] = score
    
    def _train_models(self):
        xg_dict = {'n_estimators': 200,
                        'max_depth': 65,
                        'grow_policy': 'lossguide',
                        'learning_rate': 0.09}

        rf_dict = {'n_estimators': 500,
                        'min_samples_leaf': 5,
                        'min_samples_split': 15,
                        'min_impurity_decrease': 0.05,
                        'max_depth': 85,
                        'criterion': 'entropy'}       

        gb_dict = {'n_estimators': 400,
                        'min_samples_leaf': 5,
                        'min_samples_split': 5,
                        'min_impurity_decrease': 0.0,
                        'max_depth': 5}

        lg_dict = {'penalty': 'elasticnet',
                        'solver': 'saga',
                        'l1_ratio': 0.5,
                        'C': 1.0}

        model_list = ['xgboost', 'random forest', 'gradient boosting', 'logistic regression']  
        param_list = [xg_dict, rf_dict, gb_dict]

        for model, param_dict in zip(model_list, param_list):
            self.models[model] = self._train_model(
                self.X_train, self.y_train, model, param_dict
            )

    
    def run_mccv(self, X_data, y_data, n_iters=100, test_size=0.1):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=0
        )

        self._train_models()
        self._score_models()     
        df = pd.DataFrame(self.scores)
        print('Iteration: 0')

        for i in range(1, n_iters):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_data, y_data, test_size=test_size, random_state=i
            )
            self._train_models()
            self._score_models()
            temp_df = pd.DataFrame(self.scores)
            df = pd.concat((df, temp_df))
            print(f'Iteration: {i}')

        export_df(df, 'mccv_export')
    
if __name__ == '__main__':
      main()