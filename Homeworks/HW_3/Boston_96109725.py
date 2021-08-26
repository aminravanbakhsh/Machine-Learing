import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
from sklearn.datasets import load_boston

def train_test_split(X, Y, test_percent):
    n = len(Y)
    test_size = int(test_percent * n)
    testIdx = random.sample(range(n), test_size)
    testIdx.sort()
    yTest = Y.iloc[testIdx]
    yTrain = Y.drop(testIdx)

    xTest = X.iloc[testIdx]
    xTrain = X.drop(testIdx)

    # return xtrain, xtest, ytrain, ytest
    return pd.DataFrame(xTrain), pd.DataFrame(xTest), pd.DataFrame(yTrain), pd.DataFrame(yTest)

def loss(yTest, xTest, W):
    y = yTest.to_numpy()
    x = xTest.to_numpy()
    w = np.asarray(W)

    t = np.dot(w.T,x.T)
    loss = np.dot(y.T, y) - 2 * np.dot(t, y) + np.dot(t, t.T)
    return loss[0,0]

def linearRegression(X, Y): # this function returns w
    x = X.to_numpy()
    y = Y.to_numpy()

    xt = x.T
    A = np.dot(xt, y)
    B = np.dot(xt, x)
    B_inv = np.linalg.inv(B)
    w = np.dot(B_inv, A)
    w = w.tolist()
    return w

def linearPlot(X, Y, val):
    x = X[val]
    y = Y
    plt.plot(x, y, 'bo')
    plt.xlabel(val)
    plt.ylabel('price')
    plt_title = val + ' and price'
    plt.title(plt_title)
    plt.savefig('./plot pics/' + plt_title + '.png')
    plt.show()

if __name__ == '__main__':
    boston = load_boston()  # BHP = boston house price pandas DataFrame
    BHP = pd.DataFrame(boston.data)
    BHP.columns = boston.feature_names
    BHP['price'] = boston.target
    # print(BHP.head())

    x = np.where(pd.isnull(BHP)) # empty cells location
    # print(x)

    # part 1
    X = BHP.drop('price', axis=1)
    Y = BHP['price']
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, 0.2) #split data to test and train

    # part 2
    keys = xTrain.keys()
    #### show plots and save them in plot pics directory

    for val in keys:
        # linearPlot(xTrain, yTrain, val)
        pass

    #### correlation dataframe
    cors = []
    for val in keys:
        cor = np.corrcoef(yTrain['price'], xTrain[val])
        cors.append(cor[0,1])

    corDf = pd.DataFrame({'xs': keys, 'correlation': cors})
    # print(corDf)

    #part 3 linear regression with LSTAT, DIS, and constant value
    px = xTrain[xTrain.columns]
    px['one'] = list(np.ones(xTrain.shape[0]))
    w = linearRegression(px[['one', 'LSTAT', 'DIS']], yTrain)
    # print(w)

    qx = xTest[xTest.columns]
    qx['one'] = list(np.ones(xTest.shape[0]))
    l1_test = loss(yTest, qx[['one', 'LSTAT', 'DIS']],w)
    l1_train = loss(yTrain, px[['one', 'LSTAT', 'DIS']],w)
    # print("l1_test: ", l1_test)
    # print("l1_train: ", l1_train)

    #part 4
    #### and 2rd dimention
    px = xTrain[xTrain.columns]
    keys = xTrain.keys()
    for val in keys:
        px[val + '_2'] = list(map((lambda x: x**2), xTrain[val]))

    #### correlation dataframe
    keys = px.keys()
    cors = []
    for val in keys:
        cor = np.corrcoef(yTrain['price'], px[val])
        cors.append(cor[0, 1])

    corDf = pd.DataFrame({'xs': keys, 'correlation': cors})
    # print(corDf)

    #plots
    px['one'] = list(np.ones(px.shape[0]))
    keys = px.keys()

    for val in keys:
        # linearPlot(px, yTrain, val)
        pass

    # find w
    w = linearRegression(px, yTrain)
    w_df = pd.DataFrame({'parameter': px.columns, 'w': w})
    # print(w_df)

    #loss functions
    qx = xTest[xTest.columns]
    keys = xTest.keys()
    for val in keys:
        qx[val + '_2'] = list(map((lambda x: x ** 2), xTest[val]))
    qx['one'] = list(np.ones(xTest.shape[0]))
    l2_test = loss(yTest, qx, w)
    # print("l2_test: ", l2_test)
    l2_train = loss(yTrain, px, w)
    # print("l2_train: ", l2_train)

    #part 5
    m = 10
    basis_xTrain = xTrain[xTrain.columns]
    mus_idx = np.random.choice(range(xTrain.shape[0]), m, replace=False)
    mus = xTrain.iloc[mus_idx]

    for i in range(m):
        col = []
        for j in range(xTrain.shape[0]):
            z = np.exp(-0.5 * np.linalg.norm(np.subtract(xTrain.iloc[j], mus.iloc[i])) ** 2)
            col.append(z)

        basis_xTrain['mu'+str(i+1)] = col

    #print(basis_xTrain)

    mus_keys = ['mu'+str(i+1) for i in range(m)]

    #mus correlation
    cors = []
    for val in mus_keys:
        cor = np.corrcoef(yTrain['price'], basis_xTrain[val])
        cors.append(cor[0, 1])

    corDf = pd.DataFrame({'xs': range(m), 'correlation': cors})
    # print(corDf)

    #plots
    for val in mus_keys:
        # linearPlot(basis_xTrain, yTrain, val)
        pass

    # find w
    basis_xTrain['one'] = list(np.ones(xTrain.shape[0]))
    parameter = ['mu'+str(i+1) for i in range(m)]
    parameter.append('one')
    w = linearRegression(basis_xTrain[parameter], yTrain)
    w_df = pd.DataFrame({'parameter': parameter, 'w': w})
    # print(w_df)

    #loss
    l3_train = loss(yTrain, basis_xTrain[parameter], w)
    # print("l3_train: ", l3_train)

    basis_xTest = xTest[xTest.columns]
    basis_xTest['one'] = list(np.ones(xTest.shape[0]))
    mus_idx = np.random.choice(range(xTest.shape[0]), m, replace=False)
    mus = xTest.iloc[mus_idx]

    for i in range(m):
        col = []
        for j in range(xTest.shape[0]):
            z = np.exp(-0.5 * np.linalg.norm(np.subtract(xTest.iloc[j], mus.iloc[i])) ** 2)
            col.append(z)

        basis_xTest['mu' + str(i + 1)] = col

    l3_test = loss(yTest, basis_xTest[parameter], w)
    # print("l3_test: ", l3_test)


    #part 6
    print("l1_train: ", l1_train)
    print("l1_test: ", l1_test)
    print("l1_test/l1_train", l1_test / l1_train)

    print()
    print("l2_train: ", l2_train)
    print("l2_test: ", l2_test)
    print("l2_test/l2_train", l2_test / l2_train)

    print()
    print("l3_train: ", l3_train)
    print("l3_test: ", l3_test)
    print("l3_test/l3_train", l3_test / l3_train)
