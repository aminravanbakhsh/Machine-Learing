import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import copy
from sklearn import datasets

def init():

    '''
    :return: data frame 'iris_df'
    '''

    iris_dataset = datasets.load_iris()
    iris_df = pd.DataFrame(iris_dataset.data)
    dic_column_names = {0: 'sepal length', 1: 'sepal width', 2: 'petal length', 3: 'petal width'}
    iris_df = iris_df.rename(columns=dic_column_names)
    columns_names = iris_df.keys()
    target_names = iris_dataset.target_names
    dic_target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    iris_df['target'] = iris_dataset.target
    return iris_df

def split_dataFrame(df, train_size, test_size):

    """
    :param df: dataframe
    :param train_size:
    :param test_size:
    :return: train dataframe and test dataframe
    """

    shuffle_df = df.sample(frac=1)
    shuffle_size = len(shuffle_df)
    test_df = shuffle_df[:int(shuffle_size * test_size)]
    train_df = shuffle_df[int(shuffle_size * test_size):]

    return train_df, test_df

def softmax(x):

    """
    :param x: array of data
    :return: array of softmax
    """

    m = x.max(axis = 1).reshape((x.shape[0],1))
    a = x - m
    z = np.exp(a)
    return z/np.sum(z, axis=1).reshape((x.shape[0], 1))

class perceptron:

    def __init__(self, train_df, class_type):
        self.class_type = class_type

        df = train_df.sample(frac = 1)

        self.t_train = np.array(df['target'])
        df['one'] = np.ones((len(train_df)))
        self.x_train = df.drop(columns = ['target']).to_numpy()

        self.w = np.zeros((1, self.x_train.shape[1]))

    def train(self, iteration_count, train_rate):
        '''
        :param iteration_count:
        :param train_rate:
        :return: train_w and misClassified per iteration
        '''

        misClassified_list = []

        for iteration in range(iteration_count):
            misClassified = 0
            y_train = np.dot(self.x_train, self.w.T) > 0

            for i in range(self.x_train.shape[0]):
                check = self.t_train[i] == self.class_type

                if y_train[i] ^ check:
                    misClassified += 1
                    dw = train_rate * self.x_train[i]
                    if check:
                        self.w += dw

                    else:
                        self.w -= dw

            misClassified_list.append(misClassified)

        return self.w, misClassified_list

    def predict(self, test_df):
        '''
        :param test_df:
        :return: accuracy, cofusion matrix
        '''
        df = test_df.copy()
        t_test = np.array(test_df['target'])
        df['one'] = np.ones((len(test_df)))
        x_test = df.drop(columns=['target']).to_numpy()

        misClassifieds = 0
        y_test = np.dot(x_test, self.w.T) > 0

        confusion = np.zeros((2,2))
        for i in range(x_test.shape[0]):
            check = t_test[i] == self.class_type

            if y_test[i] ^ check:
                misClassifieds += 1
                if check:
                    confusion[1][0] += 1
                else:
                    confusion[0][1] += 1

            elif check:
                confusion[1][1] += 1
            else:
                confusion[0][0] += 1
        return (1 - misClassifieds/len(test_df)), confusion

class logistic_regression:

    def __init__(self, train_df):
        df = train_df.sample(frac=1)

        self.t_train = np.array(df['target'])
        self.targets = list(sorted(df['target'].unique()))
        self.target_count = len(self.targets)
        df['one'] = np.ones((len(train_df)))
        self.x_train = df.drop(columns=['target']).to_numpy()
        self.feture_count = self.x_train.shape[1]
        self.data_count = self.x_train.shape[0]
        self.w = np.zeros((self.target_count, self.feture_count))
        self.t = np.zeros((self.data_count, self.target_count))
        for idx, j in enumerate(self.t_train):
            self.t[idx][j] = 1

    def train(self, iteration_count, train_rate, landa):
        '''
        :param iteration_count:
        :param train_rate:
        :param landa: parameter fo size of w
        :return: errors per iteration
        '''
        z = 0.999
        errors = []
        for iteration in range(iteration_count):
            h = np.dot(self.x_train, self.w.T)
            p = softmax(np.dot(self.x_train, self.w.T))
            self.w -= z * train_rate * ((np.dot((p - self.t).T, self.x_train)) + landa * self.w)
            errors.append(self._self_error())
            z *= z

        return errors

    def _self_error(self):
        p = softmax(np.dot(self.x_train, self.w.T))
        return -np.sum(self.t * np.log(p) + (1 - self.t) * np.log(1 - p))

    def error(self, test_df):
        '''
        :param test_df:
        :return: error w on test_df
        '''
        df = test_df.copy()
        t_test = np.array(df['target'])
        df['one'] = np.ones(len(test_df))
        x_test = df.drop(columns=['target']).to_numpy()
        test_data_count = x_test.shape[0]

        t = np.zeros((test_data_count, self.target_count))
        for idx, j in enumerate(t_test):
            t[idx][j] = 1

        p = softmax(np.dot(x_test, self.w.T))
        return -np.sum(t * np.log(p) + (1 - t) * np.log(1- p))

    def predict(self, test_df):
        '''
        :param test_df:
        :return: accuracy of correct classified and confusion matrix
        '''
        df = test_df.copy()
        t_test = np.array(df['target'])
        df['one'] = np.ones(len(test_df))
        x_test = df.drop(columns=['target']).to_numpy()
        test_data_count = x_test.shape[0]

        t = np.zeros((test_data_count, self.target_count))
        for idx, j in enumerate(t_test):
            t[idx][j] = 1

        p = softmax(np.dot(x_test, self.w.T))
        max_index = p.argmax(axis=1)
        confusion = np.zeros((self.target_count, self.target_count))
        misClassified = 0
        for i, j in zip(t_test, max_index):
            confusion[i][j] += 1
            if i != j:
                misClassified += 1

        accuracy = 1 - misClassified/test_data_count
        return accuracy, confusion

def main():
    iris_df = init()

    '''
        part1: perceptron
            1   split dataframe
            2   train perceptron with train_df
            3   plot misclassified per iterations
            4   test w on test_df
    '''
    # perceptron_train_iteration = 20
    # perceptron_train_rate = 0.1
    # perceptron_df = iris_df[iris_df['target'].isin([0,1])]
    # perceptron_train_df, perceptron_test_df = split_dataFrame(perceptron_df, 0.8, 0.2)
    # classifier = perceptron(perceptron_train_df, 0)
    # ws , train_miss = classifier.train(perceptron_train_iteration, perceptron_train_rate)
    # accuracy, confusion = classifier.predict(perceptron_test_df)
    #
    # plt.plot(range(perceptron_train_iteration), train_miss)
    # plt.ylabel('misclassifieds')
    # plt.xlabel('iteration')
    # plt.show()
    #
    # print(accuracy)
    # print(confusion)


    '''
        part2: logistic regression
            split dataframe
            a:
            1   train logistic regression on train_df
            2   return accuracy and confusion of test_df
            
            b:
            1   train logistic regression with regularizer on train_df
            2   return accuracy and confusion of test_df
            
    '''

    logistic_train_df, logistic_test_df = split_dataFrame(iris_df, 0.8, 0.2)
    train_iteration = 200
    learn_rate = 0.01
    '''
        part a
    '''
    classifier = logistic_regression(logistic_train_df)
    errors = classifier.train(train_iteration, learn_rate, 0)

    plt.plot(range(train_iteration), errors)
    plt.xlabel('iteration')
    plt.ylabel('log maximum likelihood')
    plt.show()

    accuracy, confusion = classifier.predict(logistic_test_df)
    print('regularizer = 0:')
    print('accuracy:',accuracy)
    print('confusion matrix:',confusion,'', sep='\n')

    '''
        part b
    '''
    classifier = logistic_regression(logistic_train_df)
    errors = classifier.train(train_iteration, learn_rate, 1)
    accuracy, confusion = classifier.predict(logistic_test_df)
    print('regularizer = 1:')
    print('accuracy:', accuracy)
    print('confusion matrix:', confusion, '', sep='\n')

    classifier = logistic_regression(logistic_train_df)
    errors = classifier.train(train_iteration, learn_rate, 10)
    accuracy, confusion = classifier.predict(logistic_test_df)
    print('regularizer = 10:')
    print('accuracy:',accuracy)
    print('confusion matrix:',confusion,'', sep='\n')

    classifier = logistic_regression(logistic_train_df)
    errors = classifier.train(train_iteration, learn_rate, 0.1)
    accuracy, confusion = classifier.predict(logistic_test_df)
    print('regularizer = 0.1:')
    print('accuracy:', accuracy)
    print('confusion matrix:', confusion,'', sep='\n')

if __name__ == '__main__':
    main()