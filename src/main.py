import torch
from src.segtime import Segtime
from src.dataset import OpportunityDS
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

BATCH_SIZE = 1
EPOCH_MAX = 20
device = 'cuda'

ds = OpportunityDS('data/Opportunity/train')
dl = DataLoader(ds,batch_size=BATCH_SIZE)

model = Segtime(113,[1,4,16,64],[2,2,2,2],5).to(device)
optim = Adam(model.parameters(),lr=0.1)
criterion = CrossEntropyLoss()

for e in range(EPOCH_MAX):
    for x,y1,y2 in dl:
        x = x.to(device)
        y1 = y1.to(device)
        output = model(x)
        loss = criterion(output,y1)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    with torch.no_grad():
        acc = 0
        nb_ex = 0
        l_g = 0
        n = 0
        for x,y1,y2 in dl:
            x = x.to(device)
            y1 = y1.to(device)
            output = model(x)
            y_hat = torch.argmax(output,dim=1)
            acc += (y_hat.view(-1)==y1).sum()
            nb_ex += x.size(1)
            loss = criterion(output,y1)
            l_g += loss
            n+=1
                
    print('loss :',(l_g/n).item(),'acc :',(acc/nb_ex).item(),'@'+str(e))