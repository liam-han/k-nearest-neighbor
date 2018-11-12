import random
import math
import numpy as np
import operator


def shuffle_data(filename):
    file = np.genfromtxt(filename, dtype=float, delimiter=',')
    file = np.delete(file, 0, 1)
    n = file[~np.isnan(file).any(axis=1)]
    random.shuffle(n)

    return n


def getData(filename):
    file = np.genfromtxt(filename, dtype=float, delimiter=',')
    file = np.delete(file, 0, 1)
    n = file[~np.isnan(file).any(axis=1)]
    data = n[0:]
    top_80 = math.floor(len(data) * .80)
    x_train = data[0:top_80]
    x_test = data[top_80 + 1:]
    y_train = ["benign", "malignant"]

    return x_train, x_test, y_train


def distance_LPnorm(X, Y, p):
    p = 2
    distance = 0
    for i in range(8):
        distance += math.pow(math.fabs(X[i] - Y[i]), p)

    new_distance = math.pow(distance, 1 / p)
    return new_distance


def neighbor(x_train, x_test, k):
    distances = []
    for x in range(len(x_train)):
        dist = distance_LPnorm(x_test, x_train, 2)
        distances.append((x_train, dist))

    distances.sort(key=operator.itemgetter(1))
    print(distances)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        # print(neighbors)
    return neighbors


# Use a dictionary to check and increment benign(2)/malignant(4) and return most occurring label
def count(neighbors, majority):
    for x in range(len(neighbors)):
        feature_class = neighbors[x][-1]
        if feature_class in majority:
            majority[feature_class] += 1
            # print(classVotes)
        else:
            majority[feature_class] = 1
            # print(classVotes)
    sorted_majority = sorted(majority.items())
    # print (classVotes)

    return sorted_majority[0][0]


def calculate_accuracy(x_test, pred):
    counter = 0
    l = len(x_test)
    for x in range(len(x_test)):
        if pred[x] == x_test[x][-1]:
            counter += 1
    accuracy = (counter / float(l) * 100.0)

    return accuracy


def knn(x_train, x_test, y_train, k, p):
    counter = 0
    dictt = dict()
    y_pred = []
    for x in range(len(x_test)):
        neighbors = neighbor(x_train[x], x_test[x], k)
        result = count(neighbors, dictt)
        y_pred.append(result)
    # print(neighbors)
    # print (y_pred)

    for x in range(len(x_test)):
        if x_train[x][-1] == 2.0:
            label = y_train[0]

        else:
            label = y_train[1]

        if result == 2.0:
            label2 = y_train[0]
        else:
            label2 = y_train[1]

        print("y_pred= " + "[" + str(label2) + "]" + " " + "actual= " + "[" + str(label) + "]")
        counter += 1
    # print (counter)
    accuracy = calculate_accuracy(x_test, y_pred)
    return accuracy
    # u = shuffle_data(filename)
    # print(u)


def main():
    filename = 'breast-cancer-wisconsin.data'
    d = getData(filename)
    x_train = d[0]
    x_test = d[1]
    y_train = d[2]
    y_pred = knn(x_train, x_test, y_train, 1, 2)
    print(str(y_pred) + '%')


if __name__ == '__main__':
    main()
