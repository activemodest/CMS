import torch
from torch.utils.data import Dataset, DataLoader
from model_4 import Net
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from test import valid
from test import test

plt.switch_backend('agg')
import random


class CustomDataset(Dataset):
    def __init__(self, statement, statement_len, meta, history, lable):
        self.lable = lable
        self.statement = statement
        self.statement_len = statement_len
        self.meta = meta
        self.history = history

    def __getitem__(self, index):
        lable_index = self.lable[index]
        statement_index = self.statement[index]
        statement_len_index = self.statement_len[index]
        meta_index = self.meta[index]
        history_index = self.history[index]
        return statement_index, statement_len_index, meta_index, history_index, lable_index

    def __len__(self):
        return len(self.lable)


# def train(train_samples,valid_samples,test_samples,index,word2vec_preweight,vocabulary_dim,process_file,jpg_file,save_model_file,batch_size=64,lr=0.0001,weight_decay=1e-4,EPOCH=40,transformer_num_layers=2,num_heads=2):
def train(train_samples, valid_samples, test_samples, index, word2vec_preweight, vocabulary_dim, process_file, jpg_file,
          save_model_file,test_label_file, para_dict):
    # print ('Traing begin')
    # print ('Prepare train data')
    train_loss_list = []
    valid_loss_list = []
    valid_acc_list = []
    best_valid_acc = 0
    best_test_acc = 0

    # train_data
    train_statement_data = [x[1] for x in train_samples]
    train_statement_data = np.array(train_statement_data)
    train_statement_data = torch.from_numpy(train_statement_data).cuda()  

    train_statement_len = [x[2] for x in train_samples]
    train_statement_len = np.array(train_statement_len)
    train_statement_len = torch.from_numpy(train_statement_len).int().cuda()  
    # train_statement_len = train_statement_len.unsqueeze(1)

    train_meta_data = [x[3] for x in train_samples]
    train_meta_data = np.array(train_meta_data)
    train_meta_data = torch.from_numpy(train_meta_data).cuda()  

    train_history_data = [x[4] for x in train_samples]
    train_history_data = np.array(train_history_data)
    train_history_data = torch.from_numpy(train_history_data)
    train_history_data = train_history_data.float().cuda()  

    train_target = [x[0] for x in train_samples]
    train_target = np.array(train_target)
    train_target = torch.from_numpy(train_target).cuda()  

    # valid data
    valid_statement_data = [x[1] for x in valid_samples]
    valid_statement_data = np.array(valid_statement_data)
    valid_statement_data = torch.from_numpy(valid_statement_data).cuda()  

    valid_statement_len = [x[2] for x in valid_samples]
    valid_statement_len = np.array(valid_statement_len)
    valid_statement_len = torch.from_numpy(valid_statement_len).int().cuda()  
    # valid_statement_len = valid_statement_len.unsqueeze(1)

    valid_meta_data = [x[3] for x in valid_samples]
    valid_meta_data = np.array(valid_meta_data)
    valid_meta_data = torch.from_numpy(valid_meta_data).cuda() 

    valid_history_data = [x[4] for x in valid_samples]
    valid_history_data = np.array(valid_history_data)
    valid_history_data = torch.from_numpy(valid_history_data)
    valid_history_data = valid_history_data.float().cuda()  

    valid_target = [x[0] for x in valid_samples]
    valid_target = np.array(valid_target)

    # test data
    test_statement_data = [x[1] for x in test_samples]
    test_statement_data = np.array(test_statement_data)
    test_statement_data = torch.from_numpy(test_statement_data).cuda()  

    test_statement_len = [x[2] for x in test_samples]
    test_statement_len = np.array(test_statement_len)
    test_statement_len = torch.from_numpy(test_statement_len).int().cuda()  
    # test_statement_len = test_statement_len.unsqueeze(1)

    test_meta_data = [x[3] for x in test_samples]
    test_meta_data = np.array(test_meta_data)
    test_meta_data = torch.from_numpy(test_meta_data).cuda()  

    test_history_data = [x[4] for x in test_samples]
    test_history_data = np.array(test_history_data)
    test_history_data = torch.from_numpy(test_history_data)
    test_history_data = test_history_data.float().cuda()  

    test_target = [x[0] for x in test_samples]
    test_target = np.array(test_target)
    print('Construct network model')
    model = Net(word2vec_preweight, vocabulary_dim, index, para_dict['transformer_num_layers'], para_dict['num_heads'],
                para_dict['dropout'])  
    #print('Model Structure',model)

    # print ('Start training......')
    train_dataset = CustomDataset(train_statement_data, train_statement_len, train_meta_data, train_history_data,
                                  train_target)
    train_loader = DataLoader(train_dataset, batch_size=para_dict['batch_size'], shuffle=False, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['weight_decay'])
    loss_func = nn.CrossEntropyLoss()
    display_interval = 50

    model.train()
    model.cuda()

    for epoch in range(para_dict['EPOCH']):
        #print ('==>EPOCH:'+str(epoch)+' '+'started')
        process_file.write('==>EPOCH:' + str(epoch) + ' ' + 'started' + '\n')
        for step, (batch_statement, batch_statement_len, batch_meta, batch_history, batch_y) in enumerate(train_loader):
            batch_statement = Variable(batch_statement).cuda()  
            batch_statement_len = Variable(batch_statement_len).cuda()  
            batch_meta = Variable(batch_meta).cuda()  
            batch_history = Variable(batch_history).cuda()  
            batch_y = Variable(batch_y).cuda()  
            output = model(batch_statement, batch_statement_len, batch_meta, batch_history)
            loss = loss_func(output, batch_y)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % display_interval == 0:
                train_loss_list.append(loss.cpu().data.numpy())
                # print ('...==>Iter:'+str(step)+' '+'train_Loss='+str(loss.cpu().data.numpy()))
                process_file.write('...==>Epoch:' + str(epoch) + ' ' + 'train_Loss=' + str(loss.data.cpu().numpy()) + '\r\n')

                valid_loss, valid_acc = valid(valid_statement_data, valid_statement_len, valid_meta_data, valid_history_data,
                                              valid_target, model, loss_func)  # ------------------------
                valid_loss_list.append(valid_loss)
                valid_acc_list.append(valid_acc)
                if best_valid_acc < valid_acc:
                    best_valid_acc = valid_acc
                    test_acc = test(test_statement_data, test_statement_len, test_meta_data, test_history_data, test_target,
                                    test_label_file,model)
                    best_test_acc = test_acc

                # print('......==>Iter:' + str(step) + ' ' + 'valid_Loss=' + str(valid_loss)+' '+'valid_Acc='+str(valid_acc))
                process_file.write(
                    '......==>Epoch:' + str(epoch) + ' ' + 'valid_Loss=' + str(valid_loss) + ' ' + 'valid_Acc=' + str(
                        valid_acc) + '\r\n')

    x = range(para_dict['EPOCH']* (len(train_samples)//para_dict['batch_size']//display_interval+1))  # this is related display_interval
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.title('Loss vs epoch')
    plt.xlim(0, para_dict['EPOCH']* (len(train_samples)//para_dict['batch_size']//display_interval+1))
    plt.ylim(min(train_loss_list + valid_loss_list), max(train_loss_list + valid_loss_list))
    plt.ylabel('Loss')
    plt.xlabel('Iter')
    plt.plot(x, train_loss_list, label='train_loss')
    plt.plot(x, valid_loss_list, label='valid_loss')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.title('train vs valid')
    plt.xlim(0, para_dict['EPOCH']* (len(train_samples)//para_dict['batch_size']//display_interval+1))
    plt.ylim(min(valid_acc_list), max(valid_acc_list))
    plt.ylabel('Acc')
    plt.xlabel('Iter')
    plt.plot(x, valid_acc_list, label='valid_acc')
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(jpg_file)
    plt.close()

    # save model
    # torch.save(model,save_model_file)  # save the whole net

    return best_test_acc
