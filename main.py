import os , glob
import utilities.utilities as util
import neuralnetwork.fcn as fcn

### Path #####
data_path = './matlib'
preprocess_path = './preprocess'
text_path = './preprocess/bilinear.txt'


#### Hyper parameter #####
prediction_type = 0 #the type of prediction
num_lines_predict = fcn.num_lines_predict #the number of lines for prediction

#### read txt files as list ######
text = util.list_lines(text_path)

#### evaluate the accuracy #####
print('accuracy =', util.evaluate(text, prediction_type, num_lines_predict))
  
test01 = util.encode_text_line(text)
print(len(test01[0]))