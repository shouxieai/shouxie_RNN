from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import pickle
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import numpy as np

def cut_file(file = "new_poetry_5.txt",num = 200):
    with open(file,encoding = 'utf-8') as f :
        datas = f.read().split("\n")[:num]
    for d_index , data in enumerate(datas):
        datas[d_index] = " ".join(data)

    with open(f"poetry_{num}.txt","w",encoding = "utf-8 ") as f2:
        f2.write("\n".join(datas))


def train_vec(file = "poetry_200.txt"):
    file = open(file,encoding="utf-8")
    model = Word2Vec(sentences=LineSentence(file),vector_size=128,window=7,min_count=1,sg=1,hs=0,workers=6)
    with open("vec3.pkl","wb")  as f:
        pickle.dump([model.syn1neg,model.wv.index_to_key,model.wv.key_to_index],f)
    file.close()


class MyDataset(Dataset):
    def __init__(self,datas,w1,word_2_index):
        self.datas = datas
        self.w1 = w1
        self.word_2_index = word_2_index

    def __getitem__(self,index):
        poetry = self.datas[index]
        words_index = [self.word_2_index[i] for i in poetry]
        x = self.w1[words_index[:-1]]
        y = words_index[1:]


        return x.reshape(len(x),self.w1.shape[1]),np.array(y,dtype = np.int64)

    def __len__(self):
        return len(self.datas)

class MyModel(nn.Module):
    def __init__(self,embedding_num,hidden_num,corpus_num):
        super().__init__()
        self.U = nn.Linear(embedding_num,hidden_num)
        self.W = nn.Linear(hidden_num,hidden_num)
        self.Tanh = nn.Tanh()
        self.V = nn.Linear(hidden_num,corpus_num)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,x,a_prev,y=None):
        h1 = self.U(x)
        h2 = self.W(a_prev)

        h = h1 + h2
        th = self.Tanh(h)

        self.pre = self.V(th)

        if y is not None:
            loss = self.cross_loss(self.pre,y)

            return loss,th
        else:
            return self.pre,th


def generate_poetry(model):
    global corpus_num,word_2_index,index_2_word,w1,hidden_num,device
    w_index = int(np.random.randint(0,corpus_num,1))
    result = [index_2_word[w_index]]

    a_prev = torch.zeros((1,hidden_num),device = device)
    for i in range(23):
        w_embedding = torch.tensor(w1[w_index])
        w_embedding = w_embedding.to(device)
        pre,a_prev = model(w_embedding,a_prev)
        w_index = int(torch.argmax(pre,dim = 1))
        result.append(index_2_word[w_index])
    return "".join(result)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("poetry_200.txt",encoding =  'utf-8') as f2:
        datas = f2.read().split("\n")
    datas = [i.replace(" ","") for i in datas]

    with open("vec3.pkl","rb")  as f:
        w1 ,index_2_word, word_2_index = pickle.load(f)

    lr = 0.001
    batch_size = 10
    epoch = 1000
    embedding_num = w1.shape[1]
    hidden_num = 50
    corpus_num = len(w1)

    dataset = MyDataset(datas,w1,word_2_index)
    dataloader = DataLoader(dataset,batch_size,False)

    model = MyModel(embedding_num,hidden_num,corpus_num)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(),lr = lr)

    for e in range(epoch):

        for batch_x,batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            a_prev = torch.zeros((batch_size,hidden_num, ),device = device)
            for w_index in range(23):

                x = batch_x[:,w_index,:]
                y = batch_y[:,w_index]
                loss,a_prev = model(x,a_prev,y)
                loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            opt.zero_grad()

        # print(f"loss:{loss:.3f}")
        print(generate_poetry(model))