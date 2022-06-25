import random
import numpy as np
import pickle
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
def population_attack(X, n = 1):
        us = [np.mean(X[:,i]) for i in range(X.shape[1])]
        ss = [np.std(X[:,i]) for i in range(X.shape[1])]

        swipes = []
        for i in range(n):
                swipe = [us[i] + random.normalvariate(0, 3)*ss[i] for i in range(X.shape[1])]
                swipes.append(swipe)
        return swipes

data = unpickling("./combined_data.pkl")
X = []
y = []
for i in data:
    X += data[i]

X, y = create_sliding_window(X, [0]*len(X), 5)
X = np.array(X)
y = np.array(y)

pop_data_100 = population_attack(X, 100)
pop_data_500 = population_attack(X, 500)
pop_data_1000 = population_attack(X, 1000)
pop_data_10000 = population_attack(X, 10000)

pickling('pop_data_100.pkl', pop_data_100)
pickling('pop_data_500.pkl', pop_data_500)
pickling('pop_data_1000.pkl', pop_data_1000)
pickling('pop_data_10000.pkl', pop_data_10000)
