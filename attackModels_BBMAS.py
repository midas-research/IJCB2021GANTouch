import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
import pandas as pd
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

def attack(pkl, model):
    x = unpickling(pkl)
    x = scaler.transform(x)
    x = transformer.transform(x)
    model = unpickling(model)
    y = [0]*len(x)
    y_pred = model.predict(x)
    far, frr, hter = HTER(y, y_pred)
    '''print ("Accuracy:", accuracy_score(y, y_pred))
    print ("FAR:", far)
    print ("FRR", frr)
    print ("HTER", hter)'''
    return accuracy_score(y, y_pred), far, frr, hter

def attack3(data, model):
  data = unpickling(data)
  X = []
  y = []
  for i in data:
    X += data[i]
    y += [0]*len(data[i])
  X1, y = create_sliding_window(X, y, 5)
  X1 = np.array(X1)
  sc = StandardScaler()
  X1 = sc.fit_transform(X1)
  transformer = Normalizer().fit(X1)
  transformer.transform(X1)
  model = unpickling(model)
  y = np.array(y)
  y_pred = model.predict(X1)
  far, frr, hter = HTER(y, y_pred)
  '''print ("Accuracy:", accuracy_score(y, y_pred))
  print ("FAR:", far)
  print ("FRR", frr)
  print ("HTER", hter)'''
  return accuracy_score(y, y_pred), far, frr, hter

def attack2(X, u, user_id, model):
    y = binarize(u, user_id)
    X1, y = create_sliding_window(X.tolist(), y, 5)
    X1 = np.array(X1)
    sc = StandardScaler()
    X1 = sc.fit_transform(X1)
    transformer = Normalizer().fit(X1)
    transformer.transform(X1)
    model = unpickling(model)
    y = np.array(y)
    X1 = X1[np.where(y==1)]
    y = y[np.where(y==1)]
    y_pred = model.predict(X1)
    far, frr, hter = HTER(y, y_pred)
    '''print ("Accuracy:", accuracy_score(y, y_pred))
    print ("FAR:", far)
    print ("FRR", frr)
    print ("HTER", hter)'''
    return accuracy_score(y, y_pred), far, frr, hter

def binarize(u, id):
        ans = []
        for i in u:
                if int(i) == int(id):
                        ans.append(1)
                else:
                        ans.append(0)
        return ans

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

u = unpickling("Pickle_Files_Swipe/Ids_U.pkl")
u = np.array(u)
X = unpickling("Pickle_Files_Swipe/features_X.pkl")
X = np.array(X)

scaler = unpickling("scaler_Vanilla_phone.pkl")
transformer = unpickling("transformer_Vanilla_phone.pkl")
gender_list = parse_demographics()

file_ptr = open("Vanilla_Attack_Pop.out", "w")

models = ["SVM", "RForest", "MLP", "XGBoost"]
for model in models:
    final_acc = []
    final_far = []
    final_frr = []
    final_hter = []
    final_acc_m = []
    final_far_m = []
    final_frr_m = []
    final_hter_m = []
    final_acc_f = []
    final_far_f = []
    final_frr_f = []
    final_hter_f = []
    for i in range(1,117):
        #acc, far, frr, hter = attack3("umdaa_features.pkl",str(model)+"_"+str(i)+"Vanilla_Phone_Vanilla.pkl")
        #acc, far, frr, hter = attack3("serwadda_features.pkl",str(model)+"_"+str(i)+"Vanilla_Phone_Vanilla.pkl")
        #acc, far, frr, hter = attack3("random_attack_data.pkl",str(model)+"_"+str(i)+"Vanilla_Phone_Vanilla.pkl")
        #acc, far, frr, hter = attack3("hmog_features.pkl",str(model)+"_"+str(i)+"Vanilla_Phone_Vanilla.pkl")
        acc, far, frr, hter = attack("pop_data_10000.pkl",str(model)+"_"+str(i)+"Vanilla_Phone_Vanilla.pkl")
        final_acc.append(acc)
        final_far.append(far)
        final_frr.append(frr)
        final_hter.append(hter)

        if (gender_list[i] == 1):
            final_acc_m.append(acc)
            final_far_m.append(far)
            final_frr_m.append(frr)
            final_hter_m.append(hter)
            file_ptr.write("{},{},{},{},{}\n".format(acc,far,frr,hter,"M"))
        else:
            final_acc_f.append(acc)
            final_far_f.append(far)
            final_frr_f.append(frr)
            final_hter_f.append(hter)
            file_ptr.write("{},{},{},{},{}\n".format(acc,far,frr,hter,"F"))

    file_ptr.write("#####################"+str(model)+"##################### \n")
    file_ptr.write("Final Acc:"+str(np.mean(final_acc))+"\n")
    file_ptr.write("Final FAR:"+str(np.mean(final_far))+"\n")
    file_ptr.write("Final FRR:"+str(np.mean(final_frr))+"\n")
    file_ptr.write("Final HTER:"+str(np.mean(final_hter))+"\n")
    file_ptr.write("Accuracy Male:"+str(np.mean(final_acc_m))+"\n")
    file_ptr.write("FAR Male:"+str(np.mean(final_far_m))+"\n")
    file_ptr.write("FRR Male:"+str(np.mean(final_frr_m))+"\n")
    file_ptr.write("HTER Male:"+str(np.mean(final_hter_m))+"\n")
    file_ptr.write("ACC Female:"+str(np.mean(final_acc_f))+"\n")
    file_ptr.write("FAR Female:"+str(np.mean(final_far_f))+"\n")
    file_ptr.write("FRR Female:"+str(np.mean(final_frr_f))+"\n")
    file_ptr.write("HTER Female:"+str(np.mean(final_hter_f))+"\n")
    file_ptr.write("########################################## \n")
