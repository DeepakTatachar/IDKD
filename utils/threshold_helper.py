import numpy as np
import torch
from sklearn.metrics import roc_curve

def threshold_helper(o_dataset, i_dataset, net, device, temp, logger, val_split):
    samples_per_dataset = min(o_dataset.train_length, i_dataset.train_length)
    total_len = int(2 * (samples_per_dataset * val_split))
    labels = torch.zeros((total_len))
    gt_label = torch.zeros((total_len))
    datasets = [i_dataset, o_dataset]
    with torch.no_grad():
        for d_idx, dataset in enumerate(datasets):
            start_idx = 0
            for _, (data, _) in enumerate(dataset.val_loader):
                out = net(data.to(device)) / temp
                conf = torch.nn.functional.softmax(out, dim=1).max(1)[0]
                stop_idx = min(samples_per_dataset, start_idx + out.shape[0])
                no_samples = stop_idx - start_idx
                labels[start_idx: stop_idx] = conf.cpu()[:no_samples]
                gt_label[start_idx: stop_idx] = d_idx
                if stop_idx == samples_per_dataset:
                    break

    labels = labels.numpy()
    gt_label = gt_label.numpy()
    fpr, tpr, thresholds = roc_curve(y_true=gt_label, y_score=labels)
    best_thresh_idx = np.argmax(tpr - fpr)
    t_opt = thresholds[best_thresh_idx]
    logger.info(t_opt)
    
    return t_opt

