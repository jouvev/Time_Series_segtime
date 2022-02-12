import torch
from src.segtime import Segtime
from src.dataset import OpportunityDS
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from pandas import read_csv,concat
from accesdonnée import getDataLoad

BATCH_SIZE = 20
EPOCH_MAX = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test = True
oportunity = True
LENGHT=600
#device = 'cuda'

if oportunity:
    df = concat([read_csv('data/Opportunity/train/S3-Drill.dat',delimiter=' +',engine='python',header=None), read_csv('data/Opportunity/train/S4-Drill.dat',delimiter=' +',engine='python',header=None)],ignore_index=True)
    dftrain = df[:-24000]
    dftest = df[-24000:]
    dstrain = OpportunityDS(dftrain)
    dstest = OpportunityDS(dftest)
    dltrain = DataLoader(dstrain,batch_size=BATCH_SIZE,shuffle=True)
    dltest = DataLoader(dstest,batch_size=BATCH_SIZE,shuffle=True)
    model = Segtime(113,[1,4,16,64],[10,10,10,10],5,64,256).to(device)
else:
    dltrain = getDataLoad(subject=[1,2], recording=[1,2],lenght = LENGHT, batchsize=BATCH_SIZE)
    model = Segtime(7,[1,4,16,64],[10,10,10,10],6,64,256).to(device).double()


optim = Adam(model.parameters(),lr=0.01)
criterion = CrossEntropyLoss()

for e in range(EPOCH_MAX):
    model.train()
    for x,y1,y2 in dltrain:
        x = x.to(device)
        y = y1.to(device)
        output = model(x)
        loss = criterion(output,y.long())
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    with torch.no_grad():
        model.eval()
        acc = 0
        nb_ex = 0
        l_g = 0
        n = 0
        for x,y1,y2 in dltrain:
            x = x.to(device)
            y = y1.to(device)
            output = model(x)
            y_hat = torch.argmax(output,dim=1)
            acc += (y_hat.view(-1)==y.view(-1)).sum()
            nb_ex += x.size(1)*x.size(0)
            loss = criterion(output,y)
            l_g += loss
            n+=1
        
        acc_t = 0
        nb_ex_t = 0
        l_g_t = 0
        n_t = 1
        if test:
            for x,y1,y2 in dltest:
                x = x.to(device)
                y = y1.to(device)
                output = model(x)
                y_hat = torch.argmax(output,dim=1)
                acc_t += (y_hat.view(-1)==y.view(-1)).sum()
                nb_ex_t += x.size(1)*x.size(0)
                loss = criterion(output,y)
                l_g_t += loss
                n_t+=1
                
    print('loss train :',(l_g/n).item(),'acc :',(acc/nb_ex).item(),'@'+str(e))
    if test:
        print('loss test:',(l_g_t/n_t).item(),'acc :',(acc_t/nb_ex_t).item(),'@'+str(e))