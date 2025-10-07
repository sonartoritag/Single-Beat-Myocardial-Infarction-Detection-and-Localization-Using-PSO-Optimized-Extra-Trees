import sys
import pickle
import joblib
import random
import easygui
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
#import miceforest as mf
import sklearn.neighbors._base
from collections import Counter
import matplotlib.pyplot as plt
#from missingpy import MissForest
warnings.filterwarnings("ignore")
from sklearn.impute import KNNImputer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
#from lwoku import RANDOM_STATE, N_JOBS, VERBOSE, get_prediction
from grid_search_utils import plot_grid_search, table_grid_search
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

from Metric_Calculation import metrics
from Metric_Calculation import get_metrics
import time
from ROC_Curves import ROC_cur
from ROC_Curves import PRC_cur


if __name__ == '__main__':
    #Reading Features and Train-Test split =============================================================================
    '''filename1 = easygui.fileopenbox(multiple=True)
    Features = pd.DataFrame()
    for files in filename1:
        temp = pd.read_csv(files, sep='\t', usecols=np.r_[0:27], header=None, engine='python')
        Features = Features._append(temp)
    Features.columns = ["Rht","Qht","Sht","QRSd","RARE","QRang","SRang","QRSang","QRd","SRd","Tht","TARE","Td","(Toff-Qon)/1000","Tonang","Toffang","Tang","TTond","TToffd","TW_polarity","mean(abs(ECG))","std(abs(ECG))","ST_seg","PWFD", "MxF","Lead","CLS"]
    df = pd.DataFrame(Features)
    gbC = df.groupby(['CLS'])
    TRAIN = pd.DataFrame()
    TEST = pd.DataFrame()
    for i, C in gbC:
        gbL = C.groupby(['Lead'])
        for j, L in gbL:
            train, test = train_test_split(L, test_size=0.2)
            TRAIN = TRAIN._append(train)
            TEST = TEST._append(test)
    TRAIN.to_csv("TRAIN.csv", sep='\t', index=False)
    TEST.to_csv("TEST.csv", sep='\t', index=False)
    print('Done')'''

    # Data Imputation===================================================================================================
    '''Train_Features = pd.read_csv("TRAIN.csv", sep='\t')
    Train_Features = Train_Features.astype({"Lead": 'int', "CLS": 'int'})
    impute = KNNImputer()  # KNN imputation
    KNNImputed = impute.fit_transform(Train_Features)
    Imputed_Train_Features = pd.DataFrame(KNNImputed, columns=Train_Features.columns)
    knnPickle = open('Knn_Imputer', 'wb')
    pickle.dump(impute, knnPickle)
    knnPickle.close()
    scaler = RobustScaler()
    cols_to_norm = ['Rht', 'Qht', 'Sht', 'QRSd', 'RARE', 'QRang', 'SRang', 'QRSang', 'QRd', 'SRd', 'Tht', 'TARE', 'Td',
                    '(Toff-Qon)/1000', 'Tonang', 'Toffang', 'Tang', 'TTond', 'TToffd', 'TW_polarity', 'mean(abs(ECG))',
                    'std(abs(ECG))', 'ST_seg', 'PWFD', 'MxF']
    Imputed_Train_Features[cols_to_norm] = pd.DataFrame(scaler.fit_transform(Imputed_Train_Features[cols_to_norm]))
    Imputed_Train_Features.to_csv('Imputed_Normalized_Train_Features.csv', sep='\t', index=False)
    scaler_filename = "Std_scaler.save"
    joblib.dump(scaler, scaler_filename)'''
    #===================================================================================================================
    '''Imputed_Normalized_Train_Features = pd.read_csv("Imputed_Normalized_Train_Features.csv", sep='\t')
    counter = Counter(Imputed_Normalized_Train_Features.iloc[:, -1])
    print('Before', counter)
    gbL = Imputed_Normalized_Train_Features.groupby(['Lead'])
    X_train_UP = pd.DataFrame()
    Y_train_UP = pd.DataFrame()
    for j, L in gbL:
        X_train = L.iloc[:, 0:-1]
        Y_train = L.iloc[:, -1]
        counter = Counter(Y_train)
        D = {21.0: int(counter[21.0]*1.5), 22.0: int(counter[22.0]*1.5), 25.0: int(counter[25.0]*1.5), 26.0: int(counter[26.0]*1.5)}
        ada = ADASYN(sampling_strategy=D, random_state=42)
        X_train_ADA, Y_train_ADA = ada.fit_resample(X_train, Y_train)
        Y_train_ADA = Y_train_ADA.to_frame()
        X_train_UP = X_train_UP._append(X_train_ADA)
        Y_train_UP = Y_train_UP._append(Y_train_ADA)

    df_concat = pd.concat([X_train_UP, Y_train_UP], axis=1)
    counter = Counter(df_concat.iloc[:, -1])
    print('After', counter)'''

    '''sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(X_train_UP, Y_train_UP)
    print(sel.estimator_.feature_importances_)'''

    from sklearn.inspection import permutation_importance
    # ... (prepare your data X, y, and train your model)
    '''model = ExtraTreesClassifier()
    model.fit(X_train_UP, Y_train_UP)
    result = permutation_importance(model, X_train_UP, Y_train_UP, n_repeats=10, random_state=42)
    importances = result.importances_mean
    cols_to_norm = ['Rht', 'Qht', 'Sht', 'QRSd', 'RARE', 'QRang', 'SRang', 'QRSang', 'QRd', 'SRd', 'Tht', 'TARE', 'Td',
                    '(Toff-Qon)/1000', 'Tonang', 'Toffang', 'Tang', 'TTond', 'TToffd', 'TW_polarity', 'mean(abs(ECG))',
                    'std(abs(ECG))', 'ST_seg', 'PWFD', 'MxF', 'Lead']
    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.bar(cols_to_norm, result.importances_mean, color='maroon', width=0.4)
    plt.show()
    columns = ['ST_seg', 'TW_polarity']
    X_train_UP.drop(columns, inplace=True, axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X_train_UP, Y_train_UP, test_size=.2, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)'''

    #xt_clf = ExtraTreesClassifier(verbose=2, random_state=3, n_jobs=-1)
    #'n_estimators': 600
    #'criterion': ['gini']
    #'max_depth': [55
    #'min_samples_split': [2
    #'min_weight_fraction_leaf': 0
    #'max_features': 13
    #'max_leaf_nodes': None
    #{'min_impurity_decrease': 0
    #'bootstrap': [False]
    #'oob_score': [False]
    #'warm_start': [True]
    '''('min_samples_split': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377],
    'min_samples_leaf': [1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377],
    'min_weight_fraction_leaf': [x / 10 for x in range(0, 10)],
    'max_features': ['auto', 'sqrt', 'log2', 2, 5, 8, 13, 21, 34, None],
    'max_leaf_nodes': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, None],
    'min_impurity_decrease': [x / 100 for x in range(0, 11)],
    'bootstrap': [True, False],
    
    'bootstrap': [True],
    'oob_score': [True, False]
    'warm_start': [True, False],
    'class_weight': ['balanced_subsample'],)'''

    #parameters = {'class_weight': ['balanced', 'balanced_subsample', None]}

    '''clf = GridSearchCV(xt_clf, parameters, cv=5) #, verbose=2, n_jobs=-1)
    clf.fit(X_train_UP, Y_train_UP)
    plot_grid_search(clf)
    print(clf.best_estimator_)'''


    '''xt_clf = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=55, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=13, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=True, class_weight='balanced_subsample', ccp_alpha=0.0, max_samples=None)
    xt_clf.fit(X_train_UP, Y_train_UP)
    filename = 'MI_Classification_from_15_ECG_leads_ExtraTreeClassifier.sav'
    pickle.dump(xt_clf, open(filename, 'wb'))'''

    '''from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train_UP, Y_train_UP)'''

    Test_Features = pd.read_csv("TEST.csv", sep='\t')
    counter = Counter(Test_Features.iloc[:, -1])
    #print('In test data', counter)
    Test_Features.columns = ["Rht", "Qht", "Sht", "QRSd", "RARE", "QRang", "SRang", "QRSang", "QRd", "SRd", "Tht",
                             "TARE", "Td", "(Toff-Qon)/1000", "Tonang", "Toffang", "Tang", "TTond", "TToffd",
                             "TW_polarity", "mean(abs(ECG))", "std(abs(ECG))", "ST_seg", "PWFD", "MxF", "Lead", "CLS"]

    imp = pickle.load(open('Knn_Imputer', 'rb'))
    Test_Features = Test_Features.astype({"Lead": 'int', "CLS": 'int'})
    #Test_Features = pd.DataFrame(imp.fit_transform(Test_Features))
    Test_Features.columns = ["Rht", "Qht", "Sht", "QRSd", "RARE", "QRang", "SRang", "QRSang", "QRd", "SRd", "Tht",
                             "TARE", "Td", "(Toff-Qon)/1000", "Tonang", "Toffang", "Tang", "TTond", "TToffd", "TW_polarity",
                             "mean(abs(ECG))", "std(abs(ECG))", "ST_seg", "PWFD", "MxF", "Lead", "CLS"]

    cols_to_norm = ['Rht', 'Qht', 'Sht', 'QRSd', 'RARE', 'QRang', 'SRang', 'QRSang', 'QRd', 'SRd', 'Tht', 'TARE', 'Td',
                    '(Toff-Qon)/1000', 'Tonang', 'Toffang', 'Tang', 'TTond', 'TToffd', 'TW_polarity', 'mean(abs(ECG))',
                    'std(abs(ECG))', 'ST_seg', 'PWFD', 'MxF']
    scaler_filename = "Std_scaler.save"
    scaler = joblib.load(scaler_filename)
    Test_Features[cols_to_norm] = scaler.fit_transform(Test_Features[cols_to_norm])
    gbL = Test_Features.groupby(['Lead'])
    '''for k, gp in gbL:
        if k[0] == 15.0:
            Test_Features = gp'''
    columns = ['ST_seg', 'TW_polarity']
    #Test_Features.drop(columns, inplace=True, axis=1)
    Test_Features = Test_Features.astype({"Lead": 'int', "CLS": 'int'})
    Test_Features = Test_Features.dropna()
    X_test = Test_Features.iloc[:, 0:-1]
    Y_test = Test_Features.iloc[:, -1]
    xt_clf = pickle.load(open('MI_Classification_from_15_ECG_leads_ExtraTreeClassifier.sav', 'rb'))

    start_time = time.time()
    Y_pred = xt_clf.predict(X_test)
    #Y_pred = clf.predict(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    print('Time / prediction = ', runtime / len(Y_pred))
    Y_test.to_csv("Y_test.csv", sep='\t', index=False)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred.to_csv("Y_pred.csv", sep='\t', index=False)
    conf_mat = confusion_matrix(Y_test, Y_pred)
    print(conf_mat)
    class_names = ['Normal', 'Anterior MI', 'Antero lateral MI', 'Antero septal MI', 'Inferior MI', 'Infero latera MI', 'Infero Postero Lateral MI']
    CR = classification_report(Y_test, Y_pred, target_names=class_names, digits=4)
    print(CR)
    MCC = matthews_corrcoef(Y_test, Y_pred)
    print('Matthew Correlation Coefficient = ', MCC)
    cm_df = pd.DataFrame(conf_mat, index=['Normal', 'A', 'AL', 'AS', 'I', 'IL', 'IPL'], columns=['Normal', 'A', 'AL', 'AS', 'I', 'IL', 'IPL'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    metrics(conf_mat)
    get_metrics(conf_mat)
    print(np.sum(conf_mat))
    plt.show()
    ROC_cur(Y_test, xt_clf.predict_proba(X_test))
    PRC_cur(Y_test, xt_clf.predict_proba(X_test))




    '''cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(xt_clf, X_train_UP, Y_train_UP, scoring='f1_weighted', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    #print('F1 Score weighted: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    y_pred = cross_val_predict(xt_clf, X_train_UP, Y_train_UP, cv=10)
    conf_mat = confusion_matrix(Y_train_UP, y_pred)
    print(conf_mat)
    CR = classification_report(Y_train_UP, y_pred)
    print(CR)'''



    '''gbL = df_concat.groupby(['Lead'])
    for j, L in gbL:
        newdf = L.drop(['Lead'], axis=1)
        ax = newdf.groupby('CLS')['QRSang'].plot(kind='kde', legend=True)
        plt.show()'''

    '''newdf = df3.drop(['Lead','CLS'], axis=1)
    gb = df3.groupby('CLS')
    fig, axes = plt.subplots(nrows=5, ncols=5)
    newdf.plot(kind='kde', subplots=True, ax=axes) #bins=int(df.shape[0]/10),
    plt.show()'''