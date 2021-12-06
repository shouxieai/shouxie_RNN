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
        self.corpus_num = corpus_num
        self.rnn = nn.RNN(input_size=embedding_num,hidden_size=hidden_num,num_layers=1,batch_first=True,bidirectional=False)
        # self.dropout = nn.Dropout(0.1)
        self.V = nn.Linear(hidden_num,corpus_num)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,x,y=None,h_0=None):

        output,h_n = self.rnn(x,h_0)
        # output = self.dropout(output)
        pre = self.V(output)

        if y is not None:
            loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),y.reshape(-1)) #
            return loss,h_n
        else:
            return pre,h_n




def generate_poetry(model):
    global corpus_num,word_2_index,index_2_word,w1,hidden_num,device
    w_index = int(np.random.randint(0,corpus_num,1))
    result = [index_2_word[w_index]]

    a_prev = torch.zeros((1,1,hidden_num),device = device)
    for i in range(1,24):
        w_embedding = torch.tensor(w1[w_index]).reshape(1,1,-1)
        w_embedding = w_embedding.to(device)
        pre,a_prev = model(w_embedding,h_0=a_prev)
        w_index = int(torch.argmax(pre,dim = -1))
        if i % 6 == 0:
            result.append("\n")
        result.append(index_2_word[w_index])
    return "".join(result)


def generate_acrostic(model):
    head_w = input("请输入: ")[:4]   # 不卑不亢
    global corpus_num, word_2_index, index_2_word, w1, hidden_num, device


    result = []

    a_prev = torch.zeros((1, 1, hidden_num), device=device)
    for i in range(4):
        if i == 0:
            w_index = word_2_index[head_w[i]]
            result.append(head_w[i])
        else:
            w_embedding = torch.tensor(w1[w_index]).reshape(1, 1, -1)
            w_embedding = w_embedding.to(device)
            _, a_prev = model(w_embedding, h_0=a_prev)
            # w_index = int(torch.argmax(pre, dim=-1))
            w_index = word_2_index[head_w[i]]
            result.append(head_w[i])

        for j in range(5):
            w_embedding = torch.tensor(w1[w_index]).reshape(1, 1, -1)
            w_embedding = w_embedding.to(device)
            pre, a_prev = model(w_embedding, h_0=a_prev) # 恭喜发财
            w_index = int(torch.argmax(pre, dim=-1))


            result.append(index_2_word[w_index])
        result.append("\n")

    return "".join(result)


if __name__ == "__main__":
    train_vec()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("poetry_200.txt",encoding =  'utf-8') as f2:
        datas = f2.read().split("\n")
    datas = [i.replace(" ","") for i in datas]

    with open("vec3.pkl","rb")  as f:
        w1 ,index_2_word, word_2_index = pickle.load(f)

    lr = 0.003
    batch_size = 3
    epoch = 60
    embedding_num = w1.shape[1]
    hidden_num = 50
    corpus_num = len(w1)

    dataset = MyDataset(datas,w1,word_2_index)
    dataloader = DataLoader(dataset,batch_size,False)

    model = MyModel(embedding_num,hidden_num,corpus_num)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)

    # model.train()
    for e in range(epoch):

        for batch_x,batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            loss,_ = model(batch_x,y = batch_y)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")
        print(generate_poetry(model))

    # model.eval()
    while True:
        try:
            print(generate_acrostic(model))
        except:
            continue