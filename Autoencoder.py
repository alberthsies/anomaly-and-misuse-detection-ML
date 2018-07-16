
# coding: utf-8

# In[67]:


get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


# In[68]:


train_data = open("NSL_KDD/20 Percent Training Set.csv", "r")
train_data = [x.rstrip().split(",") for x in train_data]


# In[69]:


for i in range(len(train_data)-1, -1, -1):
    if(train_data[i][41] != 'normal'):
        train_data.pop(i)


# In[70]:


proto = ['tcp', 'udp', 'icmp']
service = ['finger', 'nnsp', 'private', 'ftp_data', 'exec', 'http', 'smtp', 'ftp', 'other', 'eco_i', 'vmnet', 'telnet', 'domain_u', 'ctf', 'ssh', 'urp_i', 'uucp_path', 'name', 'mtp', 'ldap', 'supdup', 'discard', 'http_443', 'efs', 'IRC', 'ecr_i', 'auth', 'bgp', 'hostnames', 'iso_tsap', 'domain', 'imap4', 'echo', 'nntp', 'sunrpc', 'systat', 'csnet_ns', 'netbios_ssn', 'gopher', 'X11', 'uucp', 'whois', 'klogin', 'time', 'login', 'netbios_dgm', 'netstat', 'daytime', 'netbios_ns', 'kshell', 'Z39_50', 'link', 'printer', 'pop_2', 'ntp_u', 'courier', 'rje', 'pop_3', 'sql_net', 'remote_job', 'urh_i', 'red_i', 'shell', 'pm_dump', 'tim_i', 'http_8001', 'tftp_u', 'aol', 'http_2784', 'harvest']
flag = ['S0', 'REJ', 'SF', 'RSTO', 'S1', 'RSTR', 'S2', 'SH', 'OTH', 'RSTOS0', 'S3']


# In[71]:


test_data = open("NSL_KDD/KDDTest+.csv", "r")
test_data = [x.rstrip().split(",") for x in test_data]

train_data = [x[:41] for x in train_data]


# In[72]:


proto_dict = {}
service_dict = {}
flag_dict = {}

for i in proto:
    proto_dict[i] = []
    for x in range(len(proto)):
        if(proto[x] == i):
            proto_dict[i].append(1.0)
        else:
            proto_dict[i].append(0.0)
            
for i in service:
    service_dict[i] = []
    for x in range(len(service)):
        if(service[x] == i):
            service_dict[i].append(1.0)
        else:
            service_dict[i].append(0.0)
            
for i in flag:
    flag_dict[i] = []
    for x in range(len(flag)):
        if(flag[x] == i):
            flag_dict[i].append(1.0)
        else:
            flag_dict[i].append(0.0)
            
            
# NEED TO ALSO PREPROCESS EACH FEATURE TO BE BETWEEN 0 AND 1???


# In[79]:


new = []
for idx1, i in enumerate(train_data):
    new.append([])
    for idx2, x in enumerate(i):
        if(idx2 == 1):
            for z in proto_dict[x]:
                new[idx1].append(z)
        elif(idx2 == 2):
            for z in service_dict[x]:
                new[idx1].append(z)
        elif(idx2 == 3):
            for z in flag_dict[x]:
                new[idx1].append(z)
        else:
            new[idx1].append(float(x))


# In[80]:


n_nodes_inpl = len(new[0])
n_nodes_hl1 = 50
n_nodes_hl2 = 32
n_nodes_hl3 = 32
n_nodes_outl = len(new[0])

hidden_1_layer_vals = {
    'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))
}

hidden_2_layer_vals = {
    'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))
}

hidden_3_layer_vals = {
    'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))
}

output_layer_vals = {
    'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_outl])),
    'biases':tf.Variable(tf.random_normal([n_nodes_outl]))
}


# In[81]:


input_layer = tf.placeholder('float', [None, n_nodes_inpl])

layer_1 = tf.nn.sigmoid(
    tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),
    hidden_1_layer_vals['biases']))

layer_2 = tf.nn.sigmoid(
    tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),
    hidden_2_layer_vals['biases']))

layer_3 = tf.nn.sigmoid(
    tf.add(tf.matmul(layer_2,hidden_3_layer_vals['weights']),
    hidden_3_layer_vals['biases']))

output_layer = tf.matmul(layer_3,output_layer_vals['weights']) + output_layer_vals['biases']

output_true = tf.placeholder('float', [None, n_nodes_outl])
# define our cost function
meansq =    tf.reduce_mean(tf.square(output_layer - output_true))
# define our optimizer
learn_rate = 1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)


# In[82]:


# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# defining batch size, number of epochs and learning rate
batch_size = 100  # how many row to use together for training
hm_epochs = 500    # how many times to go through the entire dataset
tot_rows = len(train_data) # total number of data
# running the model for a 100 epochs taking 100 row in batches
# total improvement is printed out after each epoch
start_time = time.time()
for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(tot_rows/batch_size)):
        epoch_x = new[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, meansq],               feed_dict={input_layer: epoch_x,                output_true: epoch_x})
        epoch_loss += c
    if(epoch % 20 == 0):
        print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
print("Training time:", time.time() - start_time, "seconds")


# In[83]:


new_test = []
for idx1, i in enumerate(test_data):
    new_test.append([])
    for idx2, x in enumerate(i):
        if(idx2 >= 41):
            break
        if(idx2 == 1):
            for z in proto_dict[x]:
                new_test[idx1].append(z)
        elif(idx2 == 2):
            for z in service_dict[x]:
                new_test[idx1].append(z)
        elif(idx2 == 3):
            for z in flag_dict[x]:
                new_test[idx1].append(z)
        else:
            new_test[idx1].append(float(x))


# In[84]:


TP = 0
TN = 0
FP = 0
FN = 0

err_list = []

start_test = time.time()

for i in new:
    curr_err = sess.run(meansq, feed_dict={input_layer: [i], output_true: [i]})
    err_list.append(curr_err)
    
sorted_errs = sorted(err_list)
Q3_err = sorted_errs[int(len(err_list)*3/4)]
Q1_err = sorted_errs[int(len(err_list)/4)]
err_threshold = Q3_err
print(err_threshold)

# plt.plot([idx*5 for idx in range(len(err_list))], err_list, 'b.')

for idx, i in enumerate(test_data):
    if(i[41] == 'normal'):
        cr = sess.run(meansq, feed_dict={input_layer: [new_test[idx]], output_true: [new_test[idx]]})
        if(cr > err_threshold):
            # Outlier
            FP += 1
        else:
            TN += 1
    else:
        cr = sess.run(meansq, feed_dict={input_layer: [new_test[idx]], output_true: [new_test[idx]]})
        # plt.plot([5*idx+len(err_list)+1], [cr], 'r.')
        if(cr > err_threshold):
            # Outlier
            TP += 1
        else:
            FN += 1

print("Testing time:", time.time() - start_test, "seconds")

# Output the measurement
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
print("Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
print("Recall: ", TP / (TP + FN))
print("Precision: ", TP / (TP + FP))
print("False Alarm Rate: ", FP / (FP + TN))

# x_axis = np.arange(0.0, 140000.0, 1)
# plt.plot(x_axis, [err_threshold for i in range(len(x_axis))], 'k_')
# plt.show()

