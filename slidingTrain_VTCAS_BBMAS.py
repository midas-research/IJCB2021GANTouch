import time
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from ctgan import CTGANSynthesizer
from tqdm.notebook import tqdm
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from math import sqrt
from scipy.stats import gaussian_kde
from operator import itemgetter
import shutil
import math
import numpy as np
import statistics as stat
import random
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE
from authWithGAN import *
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)

def avg_utility(r):
        final = []
        for m in range(len(r[0])):
                tmp = []
                for k in range(len(r[0][0])):
                        ans = 0
                        for i in range(len(r)):
                                ans += r[i][m][k]
                        ans /= len(r)
                        tmp.append(ans)
                final.append(tmp)
        return final

def parse_demographics():
    df = pd.read_csv("./Demographics.csv", usecols=['Gender'])

    result = []
    ctr = 0
    for index, row in df.iterrows():
        if (row["Gender"] == "M"):
            result.append(1)
        else:
            result.append(0)
    return result

def HTER(y, pred):
        '''
        Params: (Expected Binary Labels)
        y: Original Labels
        pred: Predicted Labels

        Returns:
        --------------
        FAR, FRR, HTER respectively
        '''
        far = frr = 0
        for i in range(len(y)):
                if y[i] == 0 and pred[i] == 1:
                        far += 1
                if y[i] == 1 and pred[i] == 0:
                        frr += 1
        far /= len(y)
        frr /= len(y)
        hter = (frr + far)/2
        return far, frr, hter



def pickling(fname, obj):
        f = open(fname, "wb")
        pickle.dump(obj, f)
        f.close()

def unpickling(fname):
        f = open(fname, 'rb')
        g = pickle.load(f)
        f.close()
        return g

def create_sliding_window(X, Y, n):
        final_X = []
        final_Y = []
        for i in range(len(X)- n):
                temp = []
                for j in range(i, i + n):
                        temp += X[j]
                final_X.append(temp)
                final_Y.append(Y[i+n])
        return final_X, final_Y

def binarize(u, id):

        ans = []
        for i in u:
                if int(i) == int(id):
                        ans.append(1)
                else:
                        ans.append(0)
        return ans


def compare_classification(label_name, model, device, user):
        ''' Function to process the data, apply oversampling techniques (SMOTE) and run the classification model specified using GridSearchCV
        Input:  label_name: The task to be performed (Gender, Major/Minor, Typing Style)
                feature_type: The feature set to be used (Desktop, Phone, Tablet, Combined)
                top_n_features: Thu number of features to be selected using Mutual Info criterion
                model: The ML model to train and evaluate
        Output: accuracy scores, best hyperparameters of the gridsearch run'''

        user = str(user)+"Vanilla_Phone"

        if model == "SVM":
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model',SVC())
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid={'selector__k':[50, 100, 150, 200, 235] }, scoring='accuracy', return_train_score=True
                )
                #clf = SVC()
                clf.fit(X_train, y_train)

                pickling(model + '_' + str(user) + "_Vanilla.pkl", clf)

                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), 1, 1, far, frr, hter

        if model == "RForest":
                tuned_parameters = {
                        'selector__k':[50, 100, 150, 200, 235]
                }
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model',RandomForestClassifier())
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid=tuned_parameters, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)

                pickling(model + '_' + str(user) + "_Vanilla.pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), 1, 1, far, frr, hter

        if model == "XGBoost":
                tuned_parameters = {
                'selector__k': [50, 100, 150, 200, 235]
                }
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model',xgb.XGBClassifier())
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid=tuned_parameters, scoring='accuracy', return_train_score=True
                )

                clf.fit(X_train, y_train)

                pickling(model + '_' + str(user) + "_Vanilla.pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), 1, 1, far, frr, hter

        if model == "MLP":
                tuned_parameters = {
                        'selector__k':[50, 100, 150, 200, 235]
                }
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model', MLPClassifier(hidden_layer_sizes=(235,470,235),activation="relu" ,random_state=1))
                ]
                )
                clf = GridSearchCV(
                         estimator=pipeline, param_grid=tuned_parameters, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)
                pickling(model + '_' + str(user) + "_Vanilla.pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), 1, 1, far, frr, hter

def get_data(legitimate, size = 2):
  X = unpickling("Pickle_Files_Swipe/features_X.pkl")
  u = unpickling("Pickle_Files_Swipe/Ids_U.pkl")

  X, u = create_sliding_window(X, u, 5)

  X = np.array(X)
  u = np.array(u)

  data = X[np.where(u==str(legitimate))]
  labels = [1]*len(data)
  for i in range(1, 117):
    if i == legitimate:
      continue
    temp = X[np.where(u==str(i))]
    r = np.random.choice(temp.shape[0], size = size, replace=False)
    data = np.concatenate((data, temp[r, :]))
    labels += [0]*len(r)
  return data, np.array(labels)


gender_list = parse_demographics()
file_ptr = open("Vanilla_Results_MLP_Final.out", "w")
def classification_results(problem, model, device, val, res, acc_m, far_m, frr_m, hter_m, acc_f, far_f, frr_f, hter_f):
    ac, setup, vali, far, frr, hter = compare_classification(problem, model, device,val)
    if (gender_list[val] == 1):
        acc_m.append(ac)
        far_m.append(far)
        frr_m.append(frr)
        hter_m.append(hter)
        file_ptr.write("{},{},{},{},{}\n".format(ac,far,frr,hter,"M"))
    else:
        acc_f.append(ac)
        far_f.append(far)
        frr_f.append(frr)
        hter_f.append(hter)
        file_ptr.write("{},{},{},{},{}\n".format(ac,far,frr,hter,"F"))
    res.append([ac, frr, far, hter])

models = ["SVM", "RForest", "MLP", "XGBoost"]

for model in models:
    results = []
    acc_m = []
    far_m = []
    frr_m = []
    hter_m = []
    acc_f = []
    far_f = []
    frr_f = []
    hter_f = []

    for val in range(1, 117):
        #file_ptr.write("##################### User "+str(val)+" #####################\n")
        X, y = get_data(val, 3)
        X_matrix, y_vector = ADASYN(random_state=0).fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
                        X_matrix, y_vector, test_size=0.4, stratify = y_vector, random_state=0)

        X_train, X_test, y_train, y_test = train_test_split(
        X_matrix, y_vector, test_size=0.4, stratify = y_vector, random_state=42)
        pickling("X"+str(val)+".pkl", X_test)
        pickling("Y"+str(val)+".pkl", y_test)
        
        scaler = preprocessing.StandardScaler().fit(X_train)
        transformer = Normalizer().fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = transformer.transform(X_train)
        X_test = scaler.transform(X_test)
        X_test = transformer.transform(X_test)

        pickling("scaler_Vanilla_phone.pkl", scaler)
        pickling("transformer_Vanilla_phone.pkl",transformer)

        res = []
        classification_results("Authentication", model, "Phone", val, res, acc_m, far_m, frr_m, hter_m, acc_f, far_f, frr_f, hter_f)
        results.append(res)
        file_ptr.write("res: "+ str(res))

    file_ptr.write("#####################"+str(model)+"##################### \n")
    file_ptr.write("Results:"+str(avg_utility(results))+"\n")
    file_ptr.write("Accuracy Male:"+str(np.mean(acc_m))+"\n")
    file_ptr.write("FAR Male:"+str(np.mean(far_m))+"\n")
    file_ptr.write("FRR Male:"+str(np.mean(frr_m))+"\n")
    file_ptr.write("HTER Male:"+str(np.mean(hter_m))+"\n")
    file_ptr.write("ACC Female:"+str(np.mean(acc_f))+"\n")
    file_ptr.write("FAR Female:"+str(np.mean(far_f))+"\n")
    file_ptr.write("FRR Female:"+str(np.mean(frr_f))+"\n")
    file_ptr.write("HTER Female:"+str(np.mean(hter_f))+"\n")
    file_ptr.write("########################################## \n")
