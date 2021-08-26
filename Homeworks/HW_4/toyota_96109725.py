import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import copy


def wCalculator_norm1(w,x,y,alpha):
    z = y - np.dot(x, w)
    shape = z.shape

    gradient = np.zeros((w.shape[0]))
    for i in range(shape[0]):
        if z[i] > 0:
            for j in range(w.shape[0]):
                gradient[j] -= x[i][j]
        else:
            for j in range(w.shape[0]):
                gradient[j] += x[i][j]

    z = np.dot(alpha, gradient.T)/np.linalg.norm(gradient)
    wp = w.reshape((w.shape[0], 1)) - z.reshape((z.shape[0], 1))
    return wp

def wCalculator_norm2(w, x, y, alpha):

    gradeint = np.dot(x.T, np.dot(x,w) - y)
    gradeint_size = np.linalg.norm(gradeint)
    gradeint = gradeint/gradeint_size
    wp = w - np.dot(alpha, gradeint)

    return wp

def MSE(W,X,Y):
    w = np.array(list(W))
    x = np.array(list(X))
    y = np.array(list(Y))
    z = y - np.dot(x, w)
    error = np.dot(0.5, np.dot(z.T, z))
    return int(error)

def ridge(X, Y, landa):
    x = np.array(list(X))
    y = np.array(list(Y))
    xt = x.T
    shape = x.shape
    n = shape[1]
    w = np.dot(np.linalg.inv(np.dot(xt, x) + np.dot(landa,np.eye(n))), np.dot(xt, y))
    return w


if __name__ == '__main__':
    ''' collect data '''

    toyota_df = pd.read_excel("ToyotaCorolla.xls", sheet_name="data")
    df = toyota_df.drop(["Id", "Model"], axis = 1)

    ''' one hot encoding fule and color '''

    colors_df = pd.get_dummies(df["Color"])
    df = df.join(colors_df)
    fuel_types_df = pd.get_dummies(df["Fuel_Type"])
    df = df.join(fuel_types_df)

    ''' remove fuel and color column '''

    df = df.drop(columns=["Color", "Fuel_Type"])

    ''' add constant to df'''

    df['one'] = [1 for i in range(df.shape[0])]

    ''' shuffle datas ans splite train, validation and test '''

    pTrain, pValidation, pTest = 0.7, 0.15, 0.15
    df_len = len(df)
    shuffle_df = df.sample(frac=1)
    # print(shuffle_df.head())

    prices = pd.DataFrame(shuffle_df['Price'])
    y_train = prices[:int(df_len * pTrain)]
    y_validation = prices[int(df_len * pTrain): int(df_len * (pTrain + pValidation))]
    y_test = prices[int(df_len * (pTrain + pValidation)): int(df_len * (pTrain + pValidation + pTest))]

    # print(y_train.head())

    shuffle_df = shuffle_df.drop(columns=["Price"])
    x_train = shuffle_df[:int(df_len*pTrain)]
    x_validation = shuffle_df[int(df_len*pTrain): int(df_len*(pTrain + pValidation))]
    x_test = shuffle_df[int(df_len*(pTrain + pValidation)): int(df_len*(pTrain + pValidation + pTest))]

    # print(x_train.head())

    '''dfs matrix'''
    xTrain_np = x_train.to_numpy()
    yTrain_np = y_train.to_numpy()
    xValidation_np = x_validation.to_numpy()
    yValidation_np = y_validation.to_numpy()
    xTest_np = x_test.to_numpy()
    yTest_np = y_test.to_numpy()

    ''' part 1 SGD with norm 2'''
    print("part 1")

    keys = shuffle_df.keys()
    w_size = len(keys)
    w = np.array([[1] for i in range(w_size)])

    alpha = 0.1
    iteration_number = 100
    iterations = np.arange(iteration_number)
    MSE_Test = []
    MSE_Validation = []

    print("MSE test before SGD:", MSE(w,xTest_np,yTest_np))

    for i in range(iteration_number):
        w = wCalculator_norm2(w,xTrain_np,yTrain_np,alpha)
        MSE_Validation.append(MSE(w,xValidation_np,yValidation_np))
        MSE_Test.append(MSE(w,xTest_np,yTest_np))

    '''save w part 1'''
    w_part1 = w

    print("MSE test after SGD:", MSE(w,xTest_np,yTest_np))

    '''plots'''
    plt.plot(iterations, MSE_Validation)
    plt.xlabel('iteration')
    plt.ylabel('MSE_Validation')
    plt.title('MSE for validation set norm2 SGD')
    plt.savefig('MSE for validation set norm2 SGD.png')
    plt.show()

    plt.plot(iterations, MSE_Test)
    plt.xlabel('iteration')
    plt.ylabel('MSE_Test')
    plt.title('MSE for test set norm2 SGD')
    plt.savefig('MSE for test set norm2 SGD.png')
    plt.show()

    '''part 2 rigde regressor'''
    print('part 2')
    w = ridge(xTrain_np,yTrain_np,1)
    w_part2 = w
    print("MSE validation:", MSE(w, xValidation_np, yValidation_np))
    print("MSE test:", MSE(w, xTest_np, yTest_np))


    '''part 3 SGD with norm 1'''
    print("part 3")

    keys = shuffle_df.keys()
    w_size = len(keys)
    w = np.array([[1] for i in range(w_size)])

    alpha = 0.1
    iteration_number = 100
    iterations = np.arange(iteration_number)
    MSE_Test = []
    MSE_Validation = []

    print("MSE test before SGD:", MSE(w, xTest_np, yTest_np))

    for i in range(iteration_number):
        w = wCalculator_norm1(w, xTrain_np, yTrain_np, alpha)
        MSE_Validation.append(MSE(w, xValidation_np, yValidation_np))
        MSE_Test.append(MSE(w, xTest_np, yTest_np))

    '''save w part 3'''
    w_part3 = w

    print("MSE test after SGD:", MSE(w, xTest_np, yTest_np))

    '''plots'''
    plt.plot(iterations, MSE_Validation)
    plt.xlabel('iteration')
    plt.ylabel('MSE_Validation')
    plt.title('MSE for validation set norm1 SGD')
    plt.savefig('MSE for validation set norm1 SGD.png')
    plt.show()

    plt.plot(iterations, MSE_Test)
    plt.xlabel('iteration')
    plt.ylabel('MSE_Test')
    plt.title('MSE for test set norm1 SGD')
    plt.savefig('MSE for test set norm1 SGD.png')
    plt.show()


    '''part 4 compare weights'''
    print("part 4")
    w1_size = np.linalg.norm(w_part1)
    w1_nhat = w_part1/w1_size

    w3_size = np.linalg.norm(w_part3)
    w3_nhat = w_part3/w3_size

    teta_between = np.dot(w1_nhat.T, w3_nhat)
    diff = w_part3 - w_part1
    delta = 2 * np.linalg.norm(diff) / np.linalg.norm(w_part1+w_part3)

    print("different between w1 and w3:")
    print("delta:", delta, "teta between:", np.arccos(teta_between)*180 / np.pi)
    print("w1 norm2:", w1_size, "w3 norm2:", w3_size)

    w_part4 = ridge(xTrain_np, yTrain_np, 0)

    w2_size = np.linalg.norm(w_part2)
    w2_nhat = w_part2 / w2_size

    w4_size = np.linalg.norm(w_part4)
    w4_nhat = w_part4 / w4_size

    teta_between = np.dot(w2_nhat.T, w4_nhat)
    diff = w_part4 - w_part2
    delta = 2 * np.linalg.norm(diff) / np.linalg.norm(w_part2 + w_part4)

    print("different between w2 and w4:")
    print("delta:", delta, "teta between:", np.arccos(teta_between)*180 / np.pi)
    print("w2 norm2:", w2_size, "w4 norm2:", w4_size)
