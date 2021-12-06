import numpy as np


def pro_data(file="new_poetry_5.txt", nums = 50):
    with open(file,encoding = "utf-8") as f :
        datas = f.read().split("\n")[:nums]

    word_2_index = {}
    index_2_word = []

    for p in datas:
        for w in p:
            if w not in word_2_index:
                word_2_index[w] = len(word_2_index)
                index_2_word.append(w)

    return datas, word_2_index,index_2_word,np.eye(len(word_2_index))

def make_x_y(poetry):
    global word_2_index,index_2_word, wordidx_2_onehot

    w_onehot = [wordidx_2_onehot[word_2_index[i]] for i in poetry]

    x = w_onehot[:-1]
    y = w_onehot[1:]
    return x,y

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex,axis=1,keepdims=True)
    return ex/sum_ex


def generate_poetry():
    global word_2_index,index_2_word, wordidx_2_onehot,corpus_num
    w_index = int(np.random.randint(0,corpus_num,1))
    poetry_list = [index_2_word[w_index]]

    a_prev = np.zeros((1, hidden_num))
    for i in range(23):
        x_onehot = wordidx_2_onehot[w_index]
        h1 = x_onehot @ U + bias_u
        h2 = a_prev @ W + bias_w
        h = h1 + h2

        th = np.tanh(h)
        pre = th @ V + bias_v

        w_index = int(np.argmax(pre,axis = 1))
        poetry_list.append(index_2_word[w_index])

        a_prev = th

    print("".join(poetry_list))

if __name__ == "__main__":
    datas, word_2_index,index_2_word, wordidx_2_onehot = pro_data(nums=20)

    corpus_num = len(word_2_index)
    hidden_num = 107
    epoch = 1000
    lr = 0.01
    U = np.random.normal(0,1/np.sqrt(corpus_num),size=(corpus_num,hidden_num))
    W = np.random.normal(0,1/np.sqrt(hidden_num),size=(hidden_num,hidden_num))
    V = np.random.normal(0,1/np.sqrt(hidden_num),size=(hidden_num,corpus_num))
    bias_u = np.zeros((1,U.shape[1]))
    bias_w = np.zeros((1,W.shape[1]))
    bias_v = np.zeros((1,V.shape[1]))

    for e in range(epoch):
        for poetry in datas:
            x_onehots,y_onehots = make_x_y(poetry)
            a_prev = np.zeros((1, hidden_num))
            caches = []
            for x_onehot,y_onehot in zip(x_onehots,y_onehots):
                x_onehot = x_onehot[None]
                y_onehot = y_onehot[None]
                h1 = x_onehot @ U + bias_u
                h2 = a_prev @ W + bias_w
                h = h1 + h2*0.1

                th = np.tanh(h)

                pre = th @ V + bias_v
                pro = softmax(pre)

                loss = -np.sum(y_onehot * np.log(pro))
                caches.append((x_onehot,y_onehot,pro,th,a_prev))

                a_prev = th

            dth = 0

            dw = 0
            du = 0
            dv = 0



            for x_onehot,y_onehot,pro,th,a_prev in reversed(caches):
                G = pro - y_onehot
                delta_V = th.T @ G

                delta_th = G @ V.T + dth * 0.1

                delta_h = G_ = delta_th * (1-th**2)
                delta_W = a_prev.T @ delta_h
                delta_U = x_onehot.T @ delta_h

                dth = delta_h @ W.T

                # delta_bias_v = np.sum(G,axis=0,keepdims=True)
                # delta_bias_w = np.sum(G_,axis=0,keepdims=True)
                # delta_bias_u = np.sum(G_,axis=0,keepdims=True)

                dw += delta_W
                dv += delta_V
                du += delta_U

            # dw = np.clip(dw,0.0001,1)
            # dv = np.clip(dv,0.0001,1)
            # du = np.clip(du,0.0001,1)

            W -= lr * dw
            V -= lr * dv
            U -= lr * du

            # bias_u -= lr * np.mean(du,axis = 0,keepdims=True)
            # bias_w -= lr * np.mean(dw,axis = 0,keepdims=True)
            # bias_v -= lr * np.mean(dv,axis = 0,keepdims=True)

        generate_poetry()







