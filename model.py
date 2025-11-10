import torch.nn as nn
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, X, Y):
        """
        :param X: Input data is a list of numpy arrays
        :param Y: target data is a numpy arrays
        """
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        data = []
        for v_num in range(len(self.X)):
            data.append(self.X[v_num][index])
        target = self.Y[index]
        return data, target

    def __len__(self):
        return self.X[0].shape[0]


class RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(RCML, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])

    def forward(self, X):
        # get evidence
        evidences = []
        for v in range(self.num_views):
            evidences.append(self.EvidenceCollectors[v](X[v]))
        # average belief fusion
        evidence_a = evidences[0]
        r_a = 1
        for v in range(1, self.num_views):
            p_v = 1 / (1 + r_a)
            p_a = r_a / (1 + r_a)
            evidence_a = p_v * evidences[v] + p_a * evidence_a
            r_a += 1
        return evidences, evidence_a


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h
