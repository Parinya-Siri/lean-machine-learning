import os , glob
import matplotlib.pyplot as plt

def preprocess(preprocess_path, data_path):
    '''
    genereate txt files from all lean files in data_path into preprocess_path 
    
    '''
    p = os.walk(data_path)
    fps = []
    for dirpath, dirnames, filename in p:
        for f in filename:
            fp=os.path.join(dirpath,f)
            fps.append(fp)
    for filename in fps:
        save_path = preprocess_path + '/' + filename.replace(data_path, '').replace('.lean', '').replace('/','-')[1:] + '.txt'
        if os.path.exists(save_path):
            os.remove(save_path)
        open(save_path , "x")
        check = 1
        current_file = open(save_path, 'a')
        with open(filename, 'r') as f:
            for line in f:
                if '/-' in line:
                    check = 0
                if '--' in line:
                    check = 0
                if check == 1 and line != '\n':
                    current_file.write(line)
                if '-/' in line:
                    check = 1
            current_file.close()
    return fps
        
def list_lines(text_path):
    '''
    generate a list of the lines from text
    
    text_path : a string path of .txt file
    
    return : a list of lines(str) from text
    '''
    text = []
    f = open(text_path,'r')
    for line in f:
        text.append(line)
    return text



def predict(lines, prediction_type):
    '''
    Prediction function
    
    lines : a list of input lines
    prediction_type : type of prediction (integer)
    0: prediction is the last line
    
    
    return: string of prediction 
    '''
    if prediction_type == 0:
        return lines[-1]

def evaluate(text, prediction_type, num_lines_predict):
    '''
    evaluate
    
    text: 
    prediction_type: type of prediction (integer)
    
    '''
    total = 0
    correct = 0
    for i in range(len(text)-num_lines_predict-1):
        total += 1
        lines = text[i:i+num_lines_predict]
        # print(total,'th comparation between ',predict(lines, prediction_type), 'and',text[i+num_lines_predict] )
        if predict(lines, prediction_type) == text[i+num_lines_predict]:
            correct +=1
    accuracy = correct / total 
    return accuracy

def encode_text_line(text, max_char = 102):
    '''
    encode text to unicode
    
    text: a list of lines(str)
    
    return: a list of unicode for each lines with pad_num chars padding
    '''
    text_emb = []
    for line in text:
        line_text = [ord(i) for i in line]
        while len(line_text) < max_char: #padding
            line_text.insert(0,0)
        text_emb.append(line_text)
    return text_emb

def decode_text_line(text):
    '''
    decode a list of unicode to a list of str
    
    text: a list of unicode lines(int)
    
    return: a list of unicode for each lines with 80 chars padding
    '''
    de_text = []
    de_round = []
    for line in text:
        line_text = [round(max(0,i)) for i in line]
        de_round.append(line_text)
        line_text = [chr(i) for i in line_text]
        while line_text[0] == '\x00':
            del line_text[0]
        line_a = ''
        for k in line_text:
             line_a += k
        de_text.append(line_a)
    return de_text, de_round


def plot_loss(losstxtpath):
    '''
    plot loss graph from text file path losstxtpath
    '''
    losslist = open(losstxtpath, "r").readlines()
    losslist = [float(i.strip('\n')) for i in losslist]
    xpoints = range(len(losslist))
    ypoints = losslist
    plt.plot(xpoints, ypoints)