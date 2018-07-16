# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import random
import time

print("=== RF Model ===")
# Here use test dataset for training and train dataset for testing

# Loading the training dataset
train_dataset_F = open('./NSL_KDD/KDDTest+.csv', 'r')
train_dataset = []
for line in train_dataset_F:
    train_dataset.append(line.rstrip().split(","))

# Loading the testing dataset
test_dataset_F = open('./NSL_KDD/KDDTrain+.csv', 'r')
test_dataset = []
for line in test_dataset_F:
    test_dataset.append(line.rstrip().split(","))

#
# Change symbolic value to continuous value
#
proto = []
service = []
flag = []
for row in test_dataset:
    if(row[1] not in proto):
        proto.append(row[1])
    if(row[2] not in service):
        service.append(row[2])
    if(row[3] not in flag):
        flag.append(row[3])
for i in range(len(train_dataset)):
    train_dataset[i][1] = proto.index(train_dataset[i][1])
    train_dataset[i][2] = service.index(train_dataset[i][2])
    train_dataset[i][3] = flag.index(train_dataset[i][3])
for i in range(len(train_dataset)):
    for j in range(len(train_dataset[i])):
        if(j != 41):
            train_dataset[i][j] = float(train_dataset[i][j])

for i in range(len(test_dataset)):
    test_dataset[i][1] = proto.index(test_dataset[i][1])
    test_dataset[i][2] = service.index(test_dataset[i][2])
    test_dataset[i][3] = flag.index(test_dataset[i][3])
for i in range(len(test_dataset)):
    for j in range(len(test_dataset[i])):
        if(j != 41):
            test_dataset[i][j] = float(test_dataset[i][j])


# Create the NN model with 3 hidden layer (75 - 100 - 50)
# nn = MLPClassifier(hidden_layer_sizes=(75, 100, 50))

# Create Random Forest model
rf = RandomForestClassifier(random_state=0)

# X_in is the training input dataset
# X_out is the training output / target value
X_in = [x[:41] for x in train_dataset]
X_out = [x[41] for x in train_dataset]

# Train the model
# nn.fit(X_in, X_out)
print("Start training...")
start_time = time.time()
rf.fit(X_in, X_out)
print("Training duration:", time.time() - start_time, "seconds")
print()

# Y_in is the testing input dataset
# Y_out is the testing output / target value
Y_in = [y[:41] for y in test_dataset]
Y_out = [y[41] for y in test_dataset]

# Getting the list of the attacks
attacks = []
for row in train_dataset:
    if(row[41] not in attacks):
        attacks.append(row[41])
attacks.pop(attacks.index('normal'))

"""
print("=== NN Model ===")
# Calculating the Confusion Matrix
TP = 0
FN = 0
TN = 0
FP = 0
for i in range(len(Y_in)):
    out = nn.predict([Y_in[i]])
    if(out in attacks):
        if(out == Y_out[i]):
            TP += 1
        else:
            FN += 1
    else:
        if(out == Y_out[i]):
            TN += 1
        else:
            FP += 1

# Output the measurement
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
print("Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
print("Recall: ", TP / (TP + FN))
print("Precision: ", TP / (TP + FP))
print("False Alarm Rate: ", FP / (FP + TN))
"""

# Calculating the Confusion Matrix
TP = 0
FN = 0
TN = 0
FP = 0
print("Start testing...")
start_test = time.time()
for i in range(len(Y_in)):
    out = rf.predict([Y_in[i]])
    if(Y_out[i] in attacks):
        if(out == Y_out[i]):
            TP += 1
        elif(out != 'normal'):
            TP += 1
        else:
            FN += 1
    elif(Y_out[i] == 'normal'):
        if(out == Y_out[i]):
            TN += 1
        else:
            FP += 1
    else:
        FN += 1
print("Testing duration:", time.time() - start_test, "seconds")
print()

# Output the measurement
print("Results:")
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
print("Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
print("Recall: ", TP / (TP + FN))
print("Precision: ", TP / (TP + FP))
print("False Alarm Rate: ", FP / (FP + TN))
