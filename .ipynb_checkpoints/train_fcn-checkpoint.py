import torch.optim as optim
import neuralnetwork.fcn3l as fcn3
import neuralnetwork.fcn4l as fcn4
import neuralnetwork.fcn5l as fcn5
import neuralnetwork.fcn6l as fcn6
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

EPOCHS = 750
max_char = 102
num_lines_predict = 10

############ Set the Paths ######################

data_path = './matlib'
preprocess_path = './preprocess'

#### Generate txt files #########################
util.preprocess(preprocess_path,data_path)
###################Set layers##############################
layers = [int(j) for j in sys.argv[1:]]

################### Choose model ##############################
if len(sys.argv[1:]) == 3:
    net = fcn3.Net(layers, num_lines_predict, max_char)
elif len(sys.argv[1:]) == 4:
    net = fcn4.Net(layers, num_lines_predict, max_char)
elif len(sys.argv[1:]) == 5:
    net = fcn5.Net(layers, num_lines_predict, max_char)
elif len(sys.argv[1:]) == 6:
    net = fcn6.Net(layers, num_lines_predict, max_char)
    
################### Save Path ##############################
now = datetime.now()
save_path = './runs/' + now.strftime("%d-%b-%Y")
for i in range(len(layers)):
    save_path += '-'
    save_path += str(layers[i])
writer = SummaryWriter(save_path)

################### Train ##############################
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss=0
best_path=save_path + '/epoch-none.pth'
best_loss=1000
loss_list = []
print('training fcn ', layers)
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
    writer.add_scalar("Loss/train", loss, epoch)
    writer.flush()  
    
    if loss.item() < best_loss:
        if os.path.exists(best_path):
            os.remove(best_path)
        best_path = save_path + '/epoch-'+ str(epoch) + '.pth'
        torch.save(net, best_path)
        best_loss = loss
        print('saving best model')
    
    print('loss for ',epoch,' epochs : ',loss)