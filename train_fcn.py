import torch.optim as optim
import neuralnetwork.fcn as fcn
import os , glob
import utilities.utilities as util
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

############ Set the Hyper parameters ###########

EPOCHS = 10
max_char = 102
num_lines_predict = fcn.num_lines_predict
############ Set the Paths ######################

data_path = './matlib'
preprocess_path = './preprocess'

#### Generate txt files #########################
data_path = './matlib'
preprocess_path = './preprocess'
p = glob.glob(data_path + '/*', recursive = True)
for path in p:
    for filename in glob.glob(os.path.join(path, '*.lean')):
        util.preprocess(filename, preprocess_path, data_path)
    
#################################################
net = fcn.Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_list = []

for epoch in range(EPOCHS):
    print('training epoch number ',epoch)
    for filename in tqdm(glob.glob(os.path.join(preprocess_path, '*.txt'))): 
        text = util.list_lines(filename)
        en_text = util.encode_text_line(text, max_char)
        for i in range(len(text)-num_lines_predict-1):
            #choose 
            X = en_text[i:i+num_lines_predict]
            y = en_text[i+num_lines_predict]

            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)
            
            net.zero_grad()
            output = net(X.view(-1, max_char*num_lines_predict))

            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
    print('loss for ',epoch,' epochs : ',loss)
    
    
############ Save ###################
now = datetime.now()
k=0
while os.path.exists('./neuralnetwork/'+now.strftime("%m-%m-%Y")+'_'+str(k)+".txt"):
    k+=1
ppp = './neuralnetwork/'+'fcn'+now.strftime("%m-%m-%Y")+'_'+str(k)
with open(ppp+".txt", "w") as f:
    for s in loss_list:
        f.write(str(s) +"\n")
torch.save(net,ppp+'.pth')