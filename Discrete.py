import numpy as np
import matplotlib.pyplot as plt
import random
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
weights = np.ones((35, 1))# Taking the weights as ones and rate=1
rate = 1

def val(index, local_area):
    weights = []
    b = (local_area // 2)
    for i in range(index - b, index + b + 1):
        weights.append(i)
    return weights

def index(i, local_area, assoc_num, N=100):
    a_ind = local_area // 2 + ((assoc_num - 2 * (local_area // 2)) * int(i)) / N
    return math.floor(a_ind)


def rms(weights, synapse_weight, X, Y):
    rms_val= 0
    len_synpase=len(synapse_weight)
    for i in range(0, len_synpase):
        sum_synapse = 0
        for j in synapse_weight[i]:
            sum_synapse = sum_synapse + weights[j]
        rms_val += (sum_synapse - Y[i]) ** 2
    return rms_val


def Testing(weights, synapse_weight):
    output = []
    len_synpase=len(synapse_weight)
    for i in range(0, len_synpase):
        sum_synapse = 0
        for element in synapse_weight[i]:
            sum_synapse += weights[element]
        output.append(sum_synapse)
    return output


synapse = associativecells([], [])
synapse_test = associativecells([], [])


for train in X_Train:
    synapse.index = index(train, local_area, assoc_num, 100)
    synapse.weight.append(val(synapse.index, local_area))

for test in X_Test:
    synapse_test.index = (index(test, local_area, assoc_num, 100))
    synapse_test.weight.append(val(synapse_test.index, local_area))


error_list = [];error_plot = []

prevError = 0;currentError = 10
iterations = 0

while iterations < 100 and abs(prevError - currentError) > 0.001:
    prevError = currentError
    for i in range(len(synapse.weight)):
        sum_syn = 0
        for element in synapse.weight[i]:
            sum_syn += weights[element]
        error = sum_syn - Y_Train[i]
        # print(error)
        change_error = error / local_area
        # print(correction)
        for j in synapse.weight[i]:
            weights[j] -= rate * change_error
    currentError = float(rms(weights, synapse.weight, X_Train, Y_Train))
    error_list.append(currentError)
    iterations += 1
    error_plot.append(iterations)

plt.figure(2)
plt.plot(np.asarray(error_plot), np.asarray(error_list), label='error convergence')
plt.legend()
plt.show()

output = Testing(weights, synapse_test.weight)
print(output)

plt.figure(3)
plt.scatter(X_Train, Y_Train,  label='Training data')
plt.plot(X_Test, Y_Test, 'k+', label='Test data')
plt.plot(X_Test, np.asarray(output), 'go', label='outputs obtained')
plt.legend()
plt.show()

