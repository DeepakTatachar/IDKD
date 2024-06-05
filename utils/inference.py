import torch
from utils.averagemeter import AverageMeter

def inference(net, data_loader, device='cpu', loss=None):
    net.eval()
    correct = 0
    total = 0
    loss_avg = AverageMeter('Loss')
    with torch.no_grad():
        for (data, labels) in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            out = net(data)

            if(loss != None):
                loss_val = loss(out, labels)
                loss_avg.update(loss_val)

            _, pred = torch.max(out, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size()[0]
        accuracy = float(correct) * 100.0/ float(total)
    
    if(loss != None):
        return correct, total, accuracy, loss_avg.avg
    return correct, total, accuracy
