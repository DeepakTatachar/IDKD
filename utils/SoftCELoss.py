import torch

class SoftCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, logits, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = -torch.nn.functional.log_softmax(logits, dim=1)
        sample_num = target.shape[0]
        loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num
        return loss