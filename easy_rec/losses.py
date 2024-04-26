import torch

class SequentialBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        is_not_nan = ~torch.isnan(target)

        output = super().forward(input[is_not_nan], target[is_not_nan])

        return output
    
class SequentialBPR(torch.nn.Module):
    def __init__(self, eps = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, input, target):
        # Input shape: (batch_size, timesteps, num_items)
        # Output shape: (batch_size, timesteps, num_items)
        is_not_nan = ~torch.isnan(target)

        # Change relevance from 0,1 to -1,1
        new_target = target * 2 - 1
        new_target[~is_not_nan] = 0

        # Pair items in the same timestep
        item_per_relevance = (input * new_target)
        sum_per_items = item_per_relevance.sum(dim=-1)
        
        bpr = torch.log(torch.sigmoid(sum_per_items)+self.eps)

        bpr = bpr[is_not_nan.any(dim=-1)]
        
        bpr = -bpr.mean()

        return bpr