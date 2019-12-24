import torch
from torch.autograd import Variable
import re
import numpy as np

label_to_number = {
	'pants-fire': 0,
	'false': 1,
	'barely-true': 2,
	'half-true': 3,
	'mostly-true': 4,
	'true': 5
}

def meta_information_process_test(meta,word2num,meta_information):
    meta = meta.strip().split(',')
    meta_information.append(word2num.get(meta[0],0))



def test_and_valid_data_prepare(data_filename,word2num,statement_word_num=25):
    print('Prepare data from:' + data_filename)
    samples = []
    with open(data_filename, 'rb') as data_file:
        line = data_file.readline()
        while line:
            line = line.decode('utf-8')
            tmp = line.strip().split('\t')
            pre_new_item = []
            statement_information = []
            meta_information = []
            history_information = []
            while len(tmp) < 14:
                tmp.append('<no>')  # if one of the side_information no exist
            for i in range(len(tmp)):
                if i == 0:  # ID
                    pass
                elif i == 1:  # label
                    pre_new_item.append(label_to_number.get(tmp[i], 0))  # if label not exist, than make default false
                elif i == 2:  # statement
                    statement_information = re.sub('[().?!,\"\']', '',tmp[i]).strip().split()  # statement pre_process
                    if len(statement_information) < statement_word_num:
                        statement_len = len(statement_information)  # ----------------------------
                    else:
                        statement_len = statement_word_num
                    while len(statement_information) < statement_word_num:
                        statement_information.append('<no>')  
                    for j in range(len(statement_information)):  
                        statement_information[j] = word2num.get(statement_information[j],0)
                    pre_new_item.append(statement_information[0:statement_word_num])
                    pre_new_item.append(statement_len)  # ----------------------------------
                elif i > 7 and i < 13:  # history
                    history_information.append(int(tmp[i]))
                else:  # subject,speaker,speaker's job,state,party,context
                    meta_information_process_test(tmp[i], word2num, meta_information)

            pre_new_item.append(meta_information)
            pre_new_item.append(history_information)

            samples.append(pre_new_item)
            line = data_file.readline()

    return samples


def test(statement,statement_len,meta,history,target,test_label_file,model):
    # valid data
    #print('begin test')
    model.eval()
    acc = 0
    loss = 0
    out = model(statement,statement_len,meta,history,test_label=True)
    pred_y = torch.max(out, 1)[1].cpu().data.numpy()             
    with open(test_label_file,'w') as test_label_f:
        test_label_f.write('predict label for test dataset\r\n')
        for i in range(len(pred_y)):
            test_label_f.write(str(pred_y[i])+' ')
        test_label_f.write('\r\ntrue label for test dataset\r\n')
        for i in range(len(target)):
            test_label_f.write(str(target[i])+' ')
    # caculate the acc
    acc = float((pred_y == target).astype(int).sum() / len(target))
    model.train()
    return acc

def valid(statement,statement_len,meta,history,target,model,loss_func):
    model.eval()
    acc = 0
    loss = 0
    out = model(statement,statement_len,meta,history)
    pred_y = torch.max(out,1)[1].cpu().data.numpy()          

    # caculate the loss
    loss = loss_func(out, Variable(torch.from_numpy(target)).cuda())           
    # caculate the acc
    acc = float((pred_y == target).astype(int).sum() / len(target))
    model.train()
    return loss.cpu().data.numpy(), acc


