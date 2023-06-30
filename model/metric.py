import torch
from sklearn.metrics import roc_auc_score
import torchmetrics.functional as FC


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=2):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

#auroc macro average
def auroc(output, target):
    with torch.no_grad():
        num_classes = output.shape[1]
        m = torch.nn.Softmax(dim =1)
        prob = m(output)
        # print(prob)
        # print(target)

        try:
            auroc = FC.auroc(prob, target, num_classes = num_classes)
        except:
            #when auroc can not be calculated return 0
            auroc = 0 
        
        # print(auroc)
        return(auroc)
    

def confusion_matrix(output, target):
    with torch.no_grad():    
        num_classes = output.shape[1]
        pred = torch.argmax(output, dim=1)
        confusion_matrix = torch.zeros(num_classes, num_classes)
        assert pred.shape[0] == len(target)
        
        for t, p in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        return(confusion_matrix.numpy())
