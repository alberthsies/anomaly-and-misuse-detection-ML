from sklearn.ensemble import IsolationForest
import time

train_dataset_F = open("./NSL_KDD/20 Percent Training Set.csv", 'r')
# train_dataset_F = open("./NSL_KDD/KDDTrain+.csv", 'r')
train_dataset = [x.rstrip().split(",") for x in train_dataset_F]
test_dataset_F = open("./NSL_KDD/KDDTest+.csv", 'r')
test_dataset = [x.rstrip().split(",") for x in test_dataset_F]

proto = ['tcp', 'udp', 'icmp']
service = ['finger', 'nnsp', 'private', 'ftp_data', 'exec', 'http', 'smtp', 'ftp', 'other', 'eco_i', 'vmnet', 'telnet', 'domain_u', 'ctf', 'ssh', 'urp_i', 'uucp_path', 'name', 'mtp', 'ldap', 'supdup', 'discard', 'http_443', 'efs', 'IRC', 'ecr_i', 'auth', 'bgp', 'hostnames', 'iso_tsap', 'domain', 'imap4', 'echo', 'nntp', 'sunrpc', 'systat', 'csnet_ns', 'netbios_ssn', 'gopher', 'X11', 'uucp', 'whois', 'klogin', 'time', 'login', 'netbios_dgm', 'netstat', 'daytime', 'netbios_ns', 'kshell', 'Z39_50', 'link', 'printer', 'pop_2', 'ntp_u', 'courier', 'rje', 'pop_3', 'sql_net', 'remote_job', 'urh_i', 'red_i', 'shell', 'pm_dump', 'tim_i', 'http_8001', 'tftp_u', 'aol', 'http_2784', 'harvest']
flag = ['S0', 'REJ', 'SF', 'RSTO', 'S1', 'RSTR', 'S2', 'SH', 'OTH', 'RSTOS0', 'S3']

for i in range(len(train_dataset)-1, -1, -1):
    if(train_dataset[i][41] != 'normal'):
        train_dataset.pop(i)

for i in range(len(train_dataset)):
    train_dataset[i][1] = proto.index(train_dataset[i][1])
    train_dataset[i][2] = service.index(train_dataset[i][2])
    train_dataset[i][3] = flag.index(train_dataset[i][3])

for i in range(len(test_dataset)):
    test_dataset[i][1] = proto.index(test_dataset[i][1])
    test_dataset[i][2] = service.index(test_dataset[i][2])
    test_dataset[i][3] = flag.index(test_dataset[i][3])

model = IsolationForest(max_samples=50, n_estimators=200)

print("Starting training process")
start_time = time.time()
model.fit([x[:41] for x in train_dataset])
print("Training time:", time.time() - start_time, "seconds")
print()

print("Starting testing process")
test_time = time.time()
TP = 0
TN = 0
FP = 0
FN = 0

for i in test_dataset:
    result = model.predict([i[:41]])[0]
    if(result == 1 and i[41] == 'normal'):
        TN += 1
    elif(result == 1 and i[41] != 'normal'):
        FN += 1
    elif(result != 1 and i[41] == 'normal'):
        FP += 1
    else:
        TP += 1

print("Testing time:", time.time() - test_time, "seconds")
print()

# Output the measurement
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
print("Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
print("Recall: ", TP / (TP + FN))
print("Precision: ", TP / (TP + FP))
print("False Alarm Rate: ", FP / (FP + TN))
