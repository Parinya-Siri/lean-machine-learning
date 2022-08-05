import torch.optim as optim
import neuralnetwork.fcn as fcn
import os , glob
import utilities.utilities as util
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys

############ Set the Hyper parameters ###########

EPOCHS = 200
max_char = 102
num_lines_predict = 10
layers = [int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])]

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
net = fcn.Net(num_lines_predict, layers[0], layers[1], layers[2])
now = datetime.now()
save_path = './runs/' + now.strftime("%m-%m-%Y") + '-' + str(layers[0]) + '-' + str(layers[1]) + '-' + str(layers[2])
writer = SummaryWriter(save_path)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss=0
loss_list = []
print('training fcn ', layers)
for epoch in range(EPOCHS):
    print('training epoch number ',epoch)
    text = util.list_lines("./preprocess/algebra-associated.txt")
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
    writer.add_scalar("Loss/train", loss, epoch)
    writer.flush()  
    torch.save(net, save_path + '/epoch-'+ epoch + '.pth')
    print('loss for ',epoch,' epochs : ',loss)
    # loss_list.append(loss.item())
  
    
############ Save ###################
# now = datetime.now()
# k=0
# while os.path.exists('./neuralnetwork/'+now.strftime("%m-%m-%Y")+'_'+str(k)+".txt"):
#     k+=1
# ppp = './neuralnetwork/'+'fcn'+now.strftime("%m-%m-%Y")+'_'+str(k)
# with open(ppp+".txt", "w") as f:
#     for s in loss_list:
#         f.write(str(s) +"\n")
# torch.save(net,ppp+'.pth')