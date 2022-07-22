import os , glob
def preprocess(filename, preprocess_path, data_path):
    '''
    A function that create txt files from .lean files and also remove comments
    
    filename : path of the input file
    preprocess_path : destination save path
    data_path : folder path that contains .lean files
    
    return : None
    '''
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
    for line in text:
        line_text = [chr(i) for i in line]
        while line_text[0] == '\x00':
            del line_text[0]
        line_a = ''
        for k in line_text:
             line_a += k
        de_text.append(line_a)
    return de_text

