import torch
from torch import nn
import torch.nn.functional as F

class MF(nn.Module):
    # Iteration counter starting at 0
    itr = 0
    def __init__(self, n_users, n_items, k=10):
        """
        :param n_users: number of rows  
        :param n_items: number of columns
        :param k: matrix factor, r, as in the paper, this is the key factor for low rank factorization 
        """
        super(MF, self).__init__()


        # These are the hyper-parameters for the low-rank factorization (corroborative filtering)
        self.k = k
        self.n_users = n_users
        self.n_items = n_items
        self.n_occu = n_occu

        # Embeddings for users and items, these are learned, as in the FUNK_SVD and many other studies
        self.user = nn.Embedding(n_users, k)
        self.item = nn.Embedding(n_items, k)


        # Embeddings for user and item biases 
        self.bias_user = nn.Embedding(n_users, 1)
        self.bias_item = nn.Embedding(n_items, 1)

        # Bias is added for each entry, useful to shift the values, say from 0 to 100 
        self.bias = nn.Parameter(torch.ones(1))

    def __call__(self, train_x):
        """Implementing training forward and backward prop"""
        # Load user_ids, e.g., row_ids
        user_id = train_x[:, 0]
        # Load item_ids, e.g., column_ids
        item_id = train_x[:, 1]

        # Using parameter mapping, achieving Pi
        vector_user = self.user(user_id)
        # Using parameter mapping, achieving Qj 
        vector_item = self.item(item_id)

        # P[i,:] * Q[:,j]
        ui_interaction = torch.sum(vector_user * vector_item, dim=1)

        # Biases for user, iterm, and the scalar bias for each entry
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)

        # Here the example is only user-item interaction, 
        # In the toxicogenomics case, other cases can be added: tissue-drug interaction, tissue-dosage interaction
        # drug-gene interaction, drug-group interaction, etc.
        #prediction = ui_interaction + uo_interaction + biases
        prediction = ui_interaction +  biases
        return prediction

    def loss(self, prediction, target):
        """
        various losses
        """
        # MSE losses
        loss_mse = F.mse_loss(prediction.squeeze(), target.squeeze())

        #other losses that can be considered  include
        # MAE loss
        # penalized loss according to rare signals
        # general loss + rare loss
        # or only rare loss
        # or only loss for certain tissues
        # or prioritized losses for certain tissues

        total = loss_mse 

        return total
    def loss_max(self, prediction, target):
        prediction = prediction.squeeze()
        target = target.squeeze()
        error = (prediction - target) * (prediction - target)
        return torch.max(error)

    def loss_weighted(self, prediction, target):
        prediction = prediction.squeeze()
        target = target.squeeze()
        mean = self.mean
        block_size = self.block_size
        nblocks =self.nblocks
        with torch.no_grad():
            penalty_factor = weighted_penalty(self.factor_rarewrong, target, mean, block_size, nblocks)
        vect = (target - prediction)*(target - prediction)*penalty_factor
        total = vect.sum()/vect.size(0)
        return total

    def loss_step_weighted(self, prediction, target, threshold): # in here if a predict is within +/- 25% of target, we treat it as 0
        prediction = prediction.squeeze()
        target = target.squeeze()
        mean = self.mean
        block_size = self.block_size
        nblocks =self.nblocks
        with torch.no_grad():
            penalty_factor = weighted_penalty(self.factor_rarewrong, target, mean, block_size, nblocks)
        vect = (target - prediction)*(target - prediction)* (torch.abs(target-prediction) > torch.abs(target)*threshold).float() *  penalty_factor
        total = vect.sum()/vect.size(0)
        return total


def l2_regularize(array):
    """
    l2 regularization
    """
    loss = torch.sum(array ** 2.0)
    return loss
