

import csv
import math
import operator

from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from decimal import *

def getDataset(filenameTrain, filenameTest, trainingList=[] , testList=[], XSKLList=[], YSKLList=[], SKTestList=[]):
    with open(filenameTrain, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            lst = []
            lstX = []
            for y in range(len(dataset[x])):
                val1 = float(dataset[x][y]) if str(dataset[x][y]).find('.') >= 0 else int(dataset[x][y])
                if y == len(dataset[x]) - 1:
                    YSKLList.append(val1)
                else:
                    if y < 4:
                        lstX.append(val1)
                lst.append(val1)
            trainingList.append(lst)
            XSKLList.append(lstX)

    with open(filenameTest, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            lst = []
            lstSK = []
            for y in range(len(dataset[x])):
                val1 = float(dataset[x][y]) if str(dataset[x][y]).find('.') >= 0 else int(dataset[x][y])
                if y != len(dataset[x]) - 1:
                    lstSK.append(val1)
                lst.append(val1)
            testList.append(lst)
            SKTestList.append(lstSK)

def euclideanDistance(mainInstance, testInstance, length):
    distance = 0
    for x in range(length):
        distance += Decimal(pow((mainInstance[x] - testInstance[x]), 2))
    return math.sqrt(distance)

def manhattanDistance(mainInstance, testInstance, length):
    distance = 0
    for x in range(length):
        distance += Decimal(abs(mainInstance[x] - testInstance[x]))
    return distance

def getKNeighbors(trainingList, testInstance, k, type):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingList)):
        if type == 1:
            dist = euclideanDistance(testInstance, trainingList[x], length)
        else:
            dist = manhattanDistance(testInstance, trainingList[x], length)
        distances.append((trainingList[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getCorrect(testList, predictions):
    correct = 0
    for x in range(len(testList)):
        if testList[x][-1] is predictions[x]:
            correct += 1
    return correct

def main():
    trainingList=[]
    testList=[]
    XSKLList = []
    YSKLList = []
    SKTestList = []
    filenameTrain = 'Iris_train.csv'
    filenameTest = 'Iris_test.csv'
    getDataset(filenameTrain, filenameTest, trainingList, testList, XSKLList, YSKLList, SKTestList)
    kset = [1, 3, 5, 7, 9, 11, 15]

    for k in kset:
        predictions = []
        for x in range(len(testList)):


            neighbors1 = getKNeighbors(trainingList, testList[x], k, 1) #euclid
            result = getResponse(neighbors1)
            predictions.append(result)

        correct = getCorrect(testList, predictions)
        print('Euclid, k=' + repr(k) + ', Accuracy %: ' + repr(round((correct / float(len(testList))) * 100.0,2)) + ', Error Count: ' + repr(correct) + '/' + repr(len(testList)))

        predictions = []
        for x in range(len(testList)):
            neighbors1 = getKNeighbors(trainingList, testList[x], k, 2) #manhattan
            result = getResponse(neighbors1)
            predictions.append(result)

        correct = getCorrect(testList, predictions)
        print('Manhattan, k=' + repr(k) + ', Accuracy %: ' + repr(round((correct / float(len(testList))) * 100.0, 2)) + ', Error Count: ' + repr(correct) + '/' + repr(len(testList)))

    #k = 1, Euclidean distance
    #k = 3, Euclidean distance
    #k = 5, Euclidean distance
    #k = 1, Manhattan distance
    nset = [[5, 'euclidean'], [5, 'manhattan']]

    for i in nset:
        n_neighbors = i[0]
        metric = i[1]
        X = np.array(XSKLList)
        Y = np.array(YSKLList)
        Test = np.array(SKTestList)
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', metric=metric)
        clf.fit(X, Y)

        cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red', 'red'])
        cmap_bold = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red', 'red'])
        h = .02
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.title("3-Class classification : k=" + repr(n_neighbors) + ', metric:' + metric)
        plt.axis('tight')

        plt.show()

main()