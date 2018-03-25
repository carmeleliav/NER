
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import optim
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
import collections
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys


torch.manual_seed(42)
random.seed(42)



# In[88]:


class DataUtil():

    def __init__ (self, wordvec_filepath=None, news_filepath=None):
        # Some constants
        self.DEFAULT_N_CLASSES = 10
        self.DEFAULT_N_FEATURES = 300
        # Other stuff
        self.wordvecs = None
        self.word_to_ix_map = {}
        self.n_features = 0
        self.n_tag_classes = 0
        self.n_sentences_all = 0
        self.tag_vector_map = {}
        self.tag_to_id_map={}
        self.all_X = []
        self.all_Y = []
        self.train_X= []
        self.train_Y=[]
        self.test_X=[]
        self.test_Y=[]
        self.TrainDstsSet=None
        self.TestDataSet= None
        
        
        if wordvec_filepath and news_filepath:
            self.read_and_parse_data(wordvec_filepath, news_filepath)

    def read_and_parse_data (self, wordvec_filepath, news_filepath, skip_unknown_words=False):

        # Read word vectors file and create map to match words to vectors
        self.wordvecs = pd.read_table(wordvec_filepath, sep='\t', header=None)
        self.word_to_ix_map = {}
        for ix, row in self.wordvecs.iterrows():
            self.word_to_ix_map[row[0]] = ix
        self.wordvecs = self.wordvecs.drop(self.wordvecs.columns[[0,-1]], axis=1).as_matrix()
        #print self._wordvecs.shape
        self.n_features = len(self.wordvecs[0])
        # Read in training data and create map to match tag classes to tags
        # Create tag to class index map first because we need total # classes before we can
        with open(news_filepath, 'r') as f:
            self.n_tag_classes = self.DEFAULT_N_CLASSES
            self.tag_vector_map = {}
            tag_class_id = 0
            raw_news_data = []
            raw_news_words = []
            raw_news_tags = []        

            # Process all lines in the file
            for line in f:
                line = line.strip()
                if not line:
                    raw_news_data.append( (tuple(raw_news_words), tuple(raw_news_tags)) )
                    raw_news_words = []
                    raw_news_tags = []
                    continue
                word, tag = line.split('\t')
                raw_news_words.append(word)
                raw_news_tags.append(tag)
                if tag not in self.tag_vector_map:
                    one_hot_vec = torch.zeros(self.DEFAULT_N_CLASSES)
                    one_hot_vec[tag_class_id] = 1
                    self.tag_to_id_map[tag]=tag_class_id
                    self.tag_vector_map[tag] = tuple(one_hot_vec)
                    #self.tag_vector_map[tuple(one_hot_vec)] = tag
                    tag_class_id += 1
        # Add nil class
        one_hot_vec = torch.zeros(self.DEFAULT_N_CLASSES)
        one_hot_vec[tag_class_id] = 1
        self.tag_vector_map['NIL'] = tuple(one_hot_vec)
        self.tag_to_id_map['NIL']=tag_class_id


        self.n_sentences_all = len(raw_news_data)
        print(raw_news_data[0])
        
        # Build the data as required for training
        self.all_X, self.all_Y = [], []
        unk_words = []
        for words, tags in raw_news_data:
            elem_wordvecs, elem_tags = [], []
            
            for ix in range(len(words)):
                w = words[ix]
                t = tags[ix]
                
                if w in self.word_to_ix_map:
                    elem_wordvecs.append(w)
                    elem_tags.append(t)

                # Ignore unknown words, removing from dataset
                if skip_unknown_words:
                    unk_words.append(w)
                    continue
                
                # Randomly select a 300-elem vector for unknown words
                else:
                    unk_words.append(w)
                    new_wv = torch.randn(300)
                    self.word_to_ix_map[w] = self.wordvecs.shape[0]
                    self.wordvecs = np.vstack((self.wordvecs, new_wv))
                    elem_wordvecs.append(w)
                    elem_tags.append(t)
           
            prepared_x=self.prepare_sequence(words,self.word_to_ix_map)
            prepared_y=self.prepare_sequence(tags,self.tag_to_id_map)
            self.all_X.append(prepared_x)
            self.all_Y.append(prepared_y)

        
        
        #print (self.all_X[0])
        #print(self.all_Y[0])
        self.wordvecs=np.asarray(self.wordvecs)

        return (self.all_X, self.all_Y)
    
    def prepare_sequence(self,seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        tensor = torch.LongTensor(idxs)
        return(tensor)


    def get_data (self):
        return (self.all_X, self.all_Y)

    def encode_sentence (self, seq, skip_unknown_words=False): 
        idxs = []
        for word in seq:
            if word in self.word_to_ix_map:
                idxs.append(self.word_to_ix_map[word])
            elif skip_unknown_words:
                continue  
            else:
                new_wv = torch.randn(300)
                idxs.append(new_wv)

        tensor=Variable(torch.stack(idxs, 0))
        tensor=tensor.view(-1)
        return tensor
        

    def decode_prediction_sequence (self, pred_seq):
        
        pred_tags = []
        for tag in seq:
            pred_tags.append(self.tag_to_id_map.keys()[tag_to_id_map.values().index(tag)])

        tensor=torch.LongTensor(pred_tags)
        return tensor

  




    def split(self,train,test):
        train_len= int(len(self.all_X)*train)
        self.train_X=self.all_X[0:train_len]
        self.train_Y= self.all_Y[0:train_len]
        self.test_X=self.all_X[train_len:-1]
        self.test_Y=self.all_Y[train_len:-1]
        self.TrainDstsSet= Train(self.train_X,self.train_Y)
        self.TestDataSet=Test(self.test_X,self.test_Y)
        


# In[81]:



class Train(Dataset):
  
  def __init__ (self,train_X, train_Y):
      self.train_X=train_X
      self.train_Y=train_Y
                
  def __len__(self):
      return len(self.train_X)
                
                
  def __getitem__(self, idx):
          sample = {'word': self.train_X[idx], 'tag':self.train_Y[idx] }
          return  (sample)



# In[82]:


class Test(Dataset):

    def __init__ (self,test_X, test_Y):
        self.test_X=test_X
        self.test_Y=test_Y
                  
    def __len__(self):
        return len(self.test_X)
                  
    def __getitem__(self, idx):
            sample = {'word': self.test_X[idx], 'tag':self.test_Y[idx] }
            return  (sample)



# In[83]:


class BiLSTM(nn.Module):

    def __init__(self,hidden_dim,label_size, batch_size, dropout,num_layers,reader):
        super(BiLSTM, self).__init__()
        self.reader = reader
        self.pretrained_weights=reader.wordvecs
        self.vocab_size= self.pretrained_weights.shape[0]
        self.embedding_dim= self.pretrained_weights.shape[1]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(self.pretrained_weights))
        self.lstm = nn.LSTM(self.embedding_dim,hidden_dim//2, bidirectional=True,num_layers=self.num_layers,dropout=self.dropout)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()


        
    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
            return (Variable(torch.zeros(2, 1, self.hidden_dim//2)),
                    Variable(torch.zeros(2,1, self.hidden_dim//2)))
        
    def forward(self, sentence):
        embed = self.embeddings(sentence)
        embed = embed.view(len(sentence),1, -1)
        bilstm_out, self.hidden = self.lstm(embed, self.hidden)
        y = self.hidden2label(bilstm_out.view(len(sentence), -1))
        log_probs = F.log_softmax(y,dim=1)
        _, tag_seq = torch.max(log_probs, 1)
        return log_probs,tag_seq
    


# In[84]:


get_ipython().magic('matplotlib inline')

class lets_train():
    
    def __init__(self,model):
        self.model= model
        
    def train(self):
        train_loader = torch.utils.data.DataLoader(self.model.reader.TrainDstsSet,batch_size=1, num_workers=0,shuffle=True)
        test_loader  = torch.utils.data.DataLoader(self.model.reader.TestDataSet,batch_size=1, num_workers=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001,weight_decay=1e-08)
        scheduler= lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=3)
        self.epochs= 1
        plot_every =1
        print_every = 100

        self.plot_losses = []
        self.plot_train_accuracy=[]
        self.plot_test_accuracy=[]
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(self.model,epoch, train_loader,optimizer,plot_every,print_every,scheduler)
            self.test_epoch(self.model, test_loader)
        self.show_plot(self.plot_losses,'train loss','epochs','loss')
        self.show_plot(self.plot_train_accuracy,'train accuracy','epochs','accuracy')
        self.show_plot(self.plot_test_accuracy,'test accuracy','epochs','accuracy')

    def show_plot(self,points,title,x,y):
        plt.title(title)
        plt.ylabel(y)
        plt.xlabel(x)
        plt.plot(points, color='m')
        plt.show()

        
          
    def train_epoch(self,model,epoch, data_loader, optimizer,plot_every,print_every,scheduler):
        print("\n epoch {} ouf of {}##\n".format(epoch,self.epochs))
        epoch_accuracy=0
        total_len=0
        corrects=0
        train_size = len(data_loader.dataset)
        total_corrects=0
        loss_total=0
        model.train()
        for batch_idx, sample in enumerate(data_loader):
                data, tag =sample['word'],sample['tag']
                data, tag = Variable(data), Variable(tag)
                data=data.view(-1)
                tag= tag.view(-1)
                #print(sample)
                optimizer.zero_grad()
                model.hidden = model.init_hidden()
                log_probs,tag_seq = model(data)
                loss = F.nll_loss(log_probs, tag)
                loss.backward()
                optimizer.step()
                loss = loss.data[0]/len(data)
                corrects = (tag_seq.view(tag.size()).data == tag.data).sum()
                accuracy = float(corrects)/len(data) * 100.0
                total_corrects+=corrects
                total_len+=len(data)
                loss_total += loss


                if batch_idx % print_every == 0:
                    print(
                    '\tTrain batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(batch_idx,
                                                                            train_size,
                                                                             loss, 
                                                                             accuracy,
                                                                             corrects,
                                                                         len(data)))
        scheduler.step(loss_total)              
        self.plot_losses.append(loss_total)
        epoch_accuracy =total_corrects/total_len
        self.plot_train_accuracy.append(epoch_accuracy)
        print('\nTrain set: accuracy: {}/{} ({:.0f}%)\n'.format(
        total_corrects,total_len,
        100. * epoch_accuracy))
       


    def test_epoch(self,model, data_loader):
        epoch_accuracy=0
        total_len=0
        corrects=0
        total_corrects=0
        test_size = len(data_loader.dataset)
        test_loss = 0
        print ("start testing")
        model.eval()
        for batch_idx,sample in enumerate(data_loader):
            data, tag =sample['word'],sample['tag']
            data, tag = Variable(data), Variable(tag)
            data=data.view(-1)
            tag= tag.view(-1)
            model.hidden = model.init_hidden()
            log_probs,tag_seq = model(data)


            corrects = (tag_seq.view(tag.size()).data == tag.data).sum()
            total_corrects+=corrects
            total_len+=len(data)
            test_loss += (F.nll_loss(log_probs,tag, size_average=False).data[0])/len(data)
            
        epoch_accuracy=total_corrects/total_len
        self.plot_test_accuracy.append(epoch_accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,total_corrects,total_len,
        100. * epoch_accuracy))



# In[85]:


# Read the data    if __name__ == '__main__':
if __name__ == "__main__":

    WORDVEC_FILEPATH = "https://raw.githubusercontent.com/aatkinson-old/deep-named-entity-recognition/master/wordvecs.txt"
    TAGGED_NEWS_FILEPATH = "news_tagged_data.txt"



    print (">> Initializing data...")
    reader = DataUtil(WORDVEC_FILEPATH, TAGGED_NEWS_FILEPATH)
    reader.split(0.8,0.2)
    



# In[86]:


# Train the model
print (">> Training model<<")
nermodel = BiLSTM(150,10,1,0.5,1,reader)


lets_train= lets_train(nermodel)
lets_train.train()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




