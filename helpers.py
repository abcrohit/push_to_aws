import matplotlib.pyplot as plt
def plotdist(model,Dataset,granularity=1000):
    X=Dataset.x_train
    Y=Dataset.y_train
    
    preds=model(X,0)[0][:,1]
    preds=torch.exp(preds)
    n=len(preds)
    X_plot=[i/granularity for i in range(granularity+1) ]
    Y0_plot=[0 for i in range(granularity+1)]
    Y1_plot=[0 for i in range(granularity+1)]
    for i in range(n):
        pred=preds[i]
        index=int(pred*1000)
        if(Y[i]==0):
            Y0_plot[index]+=1
        else:
            Y1_plot[index]+=1
    plt.plot(X_plot,Y0_plot,c='g')
    plt.plot(X_plot,Y1_plot,c='r',ls=':')
    plt.show()
def printepoch(output,epoch,alpha):
    print(f"aplha: {alpha} epoch {epoch}"+'-'*80)
    print("training loss on source Data: ",output[0])
    print("validation loss on target Data: ",output[1])
    print("source f1:" ,output[2])
    print("target f1:" , output[3])
    print("source AUROC:" ,output[4])
    print("target AUROC:" ,output[5])
    print("source AUPRC:",output[6])
    print("target AUPRC:",output[7])
def plot(domain,parameter,dicty):
    tag={0:"loss",1:"f1",2:"Auroc",3:"Auprc"}
    dom={0:"Source" , 1:"target"}
    figure,axis=plt.subplots(21)
    figure.set_figheight(100)
    for i,key in enumerate(dicty.keys()):
        axis[i].set_title(f"{dom[domain]} {tag[parameter]} at alpha = {key}")
        values=dicty[key]
        X=[values[epoch][2*parameter+domain].item() for epoch in range(2,26)]
        Y=range(2,26)
        axis[i].plot(Y,X)

