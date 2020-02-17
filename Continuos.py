import numpy as np
import matplotlib.pyplot as plt
import math


class associativecells:
    def __init__(self, index, weight):
        self.index = 0
        self.weight = []

Sample_Data=np.arange(100)
#print(Data)
Output=np.cos(2*np.pi*Sample_Data/(100))
#print(Output)

plt.plot(Sample_Data,Output) # Visualizing Test Data
plt.show()

Stack=np.stack((Sample_Data.T,Output.T))
Data=Stack.T

#######Training 70 Samples of Data ##############
Train_Data=Data[:70]
# print(Train_Data)

### Testing 30 Samples of the Data ########
Test_Data=Data[70:]
# print(Test_Data)
np.random.shuffle(Data)

X_Train=Train_Data[:,0]
Y_Train=Train_Data[:,1]
X_Test=Test_Data[:,0]
Y_Test=Test_Data[:,1]

plt.scatter(X_Train,Y_Train, label='Train Data') #Visualizing Trained Data
plt.legend()
plt.show()

# Visualizing Test Data
plt.scatter(X_Test,Y_Test,label='Test Data')
plt.legend()
plt.show()

local_area = 5;assoc_num = 35

weights = np.ones((35, 1));rate = 1


def val(ind, local_area=5):
    weights = []
    index = []
    weightage = []
    b = local_area / 2

    index = math.floor(ind - b)
    weightage = math.ceil(ind - b) - (ind - b)

    top_cell = []
    top_cell.append(index)
    top_cell.append(weightage)

    if weightage != 0:
        weights.append(top_cell)


    for index in range(math.ceil(ind - b), math.floor(ind + b + 1)):
        mid_cell = []
        mid_cell.append(index)
        mid_cell.append(1)
        weights.append(mid_cell)


    index = math.floor(ind + local_area / 2)
    weightage = (ind + local_area / 2) - math.floor(ind + local_area / 2)

    bottom_cell = []
    bottom_cell.append(index)
    bottom_cell.append(weightage)

    if weightage != 0:
        weights.append(bottom_cell)

    return weights


def index(i, beta, assoc_num, sample):
    i = int(i)
    a_ind = beta / 2 + ((assoc_num - 2 * (beta / 2)) * i) / sample
    return a_ind


def meanSqEr(weights, synapse_weight, X, Y):
    meansq = 0
    for i in range(0, len(synapse_weight)):
        sum_syn = 0
        for j in synapse_weight[i]:
            sum_syn = sum_syn + (weights[j[0]] * j[1])
        meansq += (sum_syn - Y[i]) ** 2
    return meansq


def Test(weights, synapse_weight):
    output = []
    for i in range(0, len(synapse_weight)):
        sum_syn = 0
        for j in synapse_weight[i]:
            sum_syn += (weights[j[0]] * j[1])
        output.append(sum_syn)
    return output


synapse = associativecells([], [])
synapse_test = associativecells([], [])



for train in X_Train:
    synapse.index = index(train, local_area, assoc_num, 100)
    synapse.weight.append(val(synapse.index, local_area))

for test in X_Test:
    synapse_test.index = (index(test, local_area, assoc_num, 100))
    synapse_test.weight.append(val(synapse_test.index, local_area))


error_list = []
error_plot = []

prevError = 0
currentError = 10
iterations = 0

while iterations < 100 and abs(prevError - currentError) > 0.001:
    prevError = currentError
    for i in range(0, len(synapse.weight)):
        sum_syn = 0
        for j in synapse.weight[i]:
            sum_syn += weights[j[0]] * j[1]
            # print(sum_syn)
        error = sum_syn - Y_Train[i]
        # print(error)
        correction = error / local_area
        # print(correction)
        for j in synapse.weight[i]:
            weights[[j[0]]] -= rate * correction * j[1]
            # print(correction)
    currentError = float(meanSqEr(weights, synapse.weight, X_Train, Y_Train))
    # print(currentError)
    error_list.append(currentError)
    iterations += 1
    error_plot.append(iterations)

plt.figure(2)
plt.plot(np.asarray(error_plot), np.asarray(error_list), label='error convergence')
plt.legend()
plt.show()



output = Test(weights, synapse_test.weight)
Accuracy = float(meanSqEr(weights, synapse_test.weight, X_Test, Y_Test))

plt.figure(3)
plt.plot(X_Train, Y_Train,  'r+',label='Training data')
plt.plot(X_Test, Y_Test, 'k+', label='Test data')
plt.plot(X_Test, np.asarray(output), 'go', label='predicted outputs')
plt.legend()
plt.show()