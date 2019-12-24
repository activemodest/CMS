import re
import gensim
import os
import numpy as np

label_to_number = {
	'pants-fire': 0,
	'false': 1,
	'barely-true': 2,
	'half-true': 3,
	'mostly-true': 4,
	'true': 5
}

#convert word to num
def count_in_vocab(dict, word):
	if word not in dict:
		dict[word] = len(dict)
		return dict[word]
	else:
		return dict[word]

def meta_information_process(meta,word2num,meta_information):
    meta = meta.strip().split(',')
    meta_information.append(count_in_vocab(word2num,meta[0]))    # get first subject

# word2num -> num2word
def numtoword_convert(word2num):
    num2word = {v:k for k,v in word2num.items()}
    return num2word


def pre_word2vec(num2word,word2vec_preweight_file):
    #load pre-word2vec GoogleNews-vectors-negative300.bin ,which trained by Google News
    word2vec_preweight = []
    if os.path.exists(word2vec_preweight_file):
        with open(word2vec_preweight_file,'rb') as f:
            word2vec = f.readline()
            #print ('word2vec:',word2vec)
            while word2vec:
                word2vec = word2vec.decode('utf-8')
                word2vec = word2vec[:-1]
                word2vec = word2vec.replace('[', '')
                word2vec = word2vec.replace(']', '')
                word2vec = word2vec.split(',')
                word2vec = list(map(np.float32,word2vec))
                word2vec_preweight.append(word2vec)
                word2vec = f.readline()
    else:
        print('==>Load pre-train model GoogleNews-vectors')
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
        with open(word2vec_preweight_file,'w') as f:
            for i in range(len(num2word)):
                try:
                    word2vec_preweight.append(gensim_model[num2word[i]])
                    f.write(str(gensim_model[num2word[i]].tolist())+'\n')
                    #f.write(gensim_model[num2word[i]])
                except:
                    # word2vec_preweight.append(gensim_model['no'])
                    # f.write(str(gensim_model['no'].tolist()) + '\n')
                    word2vec_preweight.append(np.random.normal(loc=0.0, scale=1, size=(300)))
                    f.write(str(np.random.normal(loc=0.0, scale=1, size=(300)).tolist()) + '\n')
    word2vec_preweight = np.array(word2vec_preweight)

    return word2vec_preweight
    # load GoogleNews-vectors-negative300.bin

def train_data_prepare(data_filename,word2vec_preweight_file,statement_word_num=25):
    #statement_word_num    ;the number of word in statement will be used to build word2vec
    print ('Prepare data from:'+data_filename)
    samples = []
    word2num = {'<no>':0}
    with open(data_filename,'rb') as data_file:
        line = data_file.readline()
        while line:
            line = line.decode('utf-8')
            tmp = line.strip().split('\t')
            pre_new_item = []
            statement_information = []
            meta_information = []
            history_information = []
            while len(tmp) < 14:
                tmp.append('<no>')      #if one of the side_information no exist
            for i in range(len(tmp)):
                if i == 0:   #ID
                    pass
                elif i == 1:    #label
                    pre_new_item.append(label_to_number.get(tmp[i],0))     #if label not exist, than make default false
                elif i == 2:    #statement
                    statement_information = re.sub('[().?!,\"\']', '', tmp[i]).strip().split()  # statement pre_process
                    if len(statement_information) < statement_word_num:
                        statement_len = len(statement_information)  # ----------------------------
                    else:
                        statement_len = statement_word_num
                    while len(statement_information) < statement_word_num:
                        statement_information.append('<no>')
                    for j in range(len(statement_information)):
                        statement_information[j] = count_in_vocab(word2num,statement_information[j])
                    pre_new_item.append(statement_information[0:statement_word_num])
                    pre_new_item.append(statement_len)    #----------------------------------
                elif i > 7 and i < 13:        #history
                    history_information.append(int(tmp[i]))
                else:   #subject,speaker,speaker's job,state,party,context
                    meta_information_process(tmp[i],word2num,meta_information)

            pre_new_item.append(meta_information)
            pre_new_item.append(history_information)
            samples.append(pre_new_item)
            line = data_file.readline()

    num2word = numtoword_convert(word2num)
    word2vec_preweight = pre_word2vec(num2word,word2vec_preweight_file)

    return word2num,word2vec_preweight,samples


# if __name__ == '__main__':
#     data_filename = './train.tsv'
#     word2vec_preweight_file = './word2vec_preweight_file.txt'
#     word2num,word2vec_preweight,samples = train_data_prepare(data_filename, word2vec_preweight_file)
#     print('word2num[0:20]',word2num['group'])
#     print('len(word2num):',len(word2num))
#     print('word2vec_preweight[0:20]:',word2vec_preweight[0])
#     print('len(word2vec_preweight):',len(word2vec_preweight))
#     print('samples[0:2]:',samples[0])







