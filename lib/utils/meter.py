import torch
import numpy as np

class LossMeter(object):
    def __init__(self, loss_func):
        self.running_loss = []
        self.count = 0  # count = number of all received values, including those cleared.
        self.loss_func = loss_func

    def update(self, pred, gt, get_result=False, fg_weight=None):
        self.count += 1
        if fg_weight is not None:
            loss = self.loss_func(pred, gt, weight=[1, fg_weight])
        else:
            loss = self.loss_func(pred, gt)
        self.running_loss.append(loss.detach())

        if get_result:
            return loss

    def get_metric(self):
        avg = 0
        for p in self.running_loss:
            avg += p
        loss_avg = avg*1.0 / len(self.running_loss) if len(self.running_loss)!=0 else None
        return loss_avg

    def reset(self):
        self.running_loss = []

class MultiLossMeter(object):
    def __init__(self):
        self.running_loss = {}
        self.loss_names = None
        self.count = 0

    def reset(self):
        self.count = 0
        self.running_loss = {}
        if self.loss_names is not None:
            for term in self.loss_names:
                self.running_loss[term] = 0.0

    def update(self, losses, loss_names):
        if self.loss_names is None:
            self.loss_names = loss_names
            self.reset()
        self.count += 1
        loss_terms = dict(zip(loss_names, losses))

        # update running loss
        for term in self.running_loss.keys():
            if term in loss_terms.keys():
                self.running_loss[term] += loss_terms[term].detach()

    def get_metric(self):
        keys = self.running_loss.keys()
        avg_terms = {}
        for key in keys:
            avg_terms[key] = 0.0
        for key in keys:
            avg_terms[key] = self.running_loss[key] * 1.0 / self.count

        return avg_terms



def l1_loss(pred, gt):
    return abs(pred - gt)

if __name__ == '__main__':
    loss_meter = LossMeter(l1_loss)
    data = [
        (1, 2),
        (3, 5),
        (0, 4),
    ]

    for (pred, gt) in data:
        loss = loss_meter.update(pred, gt, get_result=True)
        print(loss)
    print(loss_meter.get_metric())