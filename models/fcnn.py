"""Classes to provide .fit and .score methods for a fully-connected neural network in PyTorch.

Seong-Eun Cho. 2019-03-09.
"""


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable


class TimitMelClassifier(nn.Module):
    """Simple linear network designed to accept input of length 80 (like from the mel filter we use) and output data of
    length 61.
    """
    def __init__(self, num_layers, hidden_dim):
        """Initialize layers.

        Args:
            num_layers (int): number of layers to use in the network.
            hidden_dim (int): dimension of data passing between layers.
        """
        super(TimitMelClassifier, self).__init__()
        embedding_dim = 80
        output_dim = 61

        self.input_layer = nn.Linear(embedding_dim, hidden_dim)
        self.net = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Apply network to data.

        Args:
            x (torch.Tensor): input data (shape (batch, 80))
        Returns:
            (torch.Tensor): output data (shape (batch, 61))
        """
        out = self.relu(self.input_layer(x))
        for l in self.net:
            out = self.relu(l(out))
        out = self.relu(self.output_layer(out))
        return out


class TimitMelDataset(Dataset):
    """PyTorch dataset for handling the data."""
    def __init__(self, X, y):
        """Store the features and labels.

        Args:
            X (list-like): data features.
            y (list-like): data labels.
        """
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        """Get the length of the data.

        Returns:
            (int): length of the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """Index into both the features and the labels.

        Args:
            idx (int): index of data point to return.
        Returns:
            (object): selected entry of the features.
            (object): selected entry of the labels.
        """
        return self.X[idx], self.y[idx]

class FCNN:
    """Wrapper for TimitMelClassifier with sklearn-style API."""
    def __init__(self, num_layers=5, hidden_dim=128, batch_size=128, epochs=100, eta=1e-4):
        """Store parameters.

        Args:
            num_layers (int): number of layers to use in the network.
            hidden_dim (int): dimension of data passing between layers.
            batch_size (int): batch size to pass to network.
            epochs (int): number of epochs to train the network over.
            eta (float): learning rate for Adam optimizer.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.eta = eta
        self.model = TimitMelClassifier(num_layers, hidden_dim)
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        self.dataset = None
        self.dataloader = None

    def fit(self, X, y):
        """Load the dataset and call self._train.

        Args:
            X (list-like): data features.
            y (list-like): data labels.
        Returns:
            (FCNN): reference to self, as per sklearn API requirement.
        """
        self.dataset = TimitMelDataset(X, y)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        self._train()
        return self

    def score(self, X, y):
        """Apply the network to data and return the accuracy score.

        Args:
            X (list-like): data features.
            y (list-like): data labels.
        Returns:
            (float): accuracy score of model's predictions.
        """
        tensor = torch.from_numpy(X)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            tensor = tensor.to(device)
        else:
            device = torch.device('cpu')
        
        y_pred = self.model(Variable(tensor).float()).argmax(dim=1).detach().to('cpu')  # pylint: disable=no-member
        return sum(y_pred.numpy() == y) / len(y)

    def _train(self):
        """Train the network using cross entropy loss for the specified number of epochs."""
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        for _ in range(self.epochs):
            for x, y_truth in self.dataloader:
                x = x.to(device)
                y_truth = y_truth.to(device)
                self.optimizer.zero_grad()
                y_hat = self.model(x.float())
                loss = self.objective(y_hat, y_truth.long())
                loss.backward()
                self.optimizer.step()
