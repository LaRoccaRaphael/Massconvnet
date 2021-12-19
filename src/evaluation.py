import numpy as np
from tqdm import tqdm
import time
import torch
np.seterr(divide='ignore', invalid='ignore')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ConfusionMatrix():

	def __init__(self, num_classes):

		self.num_classes = 0
		self.matrix = np.zeros((num_classes,num_classes))

	def update(self, y_pred, y_true):
		# Update the confusion matrix
		# :param y_pred: Predictions in format (batch_size,num_classes)
		# :param y_true: grountruth in Long Tensor o, format (batch_size,)

		predicted_class = torch.argmax(y_pred, dim=-1).cpu().detach().numpy()
		y_true = y_true.cpu().detach().numpy()

		for i in np.arange(predicted_class.shape[0]):
			self.matrix[int(y_true[i]),int(predicted_class[i])] += 1

	def performance(self):

		perf = dict()
		perf["accuracy"] = self.accuracy()
		return perf

	def accuracy(self):

		return np.trace(self.matrix)/np.sum(self.matrix)

	def __str__(self):

		return str(self.matrix)