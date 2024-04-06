import torch

class SequentialBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        is_not_nan = ~torch.isnan(target)

        output = super().forward(input[is_not_nan], target[is_not_nan])

        return output
# TODO delete before pushing
# class SequentialBPR(torch.nn.Module):
#     def __init__(self, eps=1e-5):
#         super().__init__()
#         self.eps = eps

#     def forward(self, input, target):
#         # Input shape: (batch_size, timesteps, num_items)
#         # Output shape: (batch_size, timesteps, num_items)

#         # Change relevance from 0,1 to -1,1
#         new_target = target * 2 - 1

#         # Pair items in the same timestep
#         # (batch_size, timesteps, num_items, num_items)
#         item_per_relevance = (input * new_target)
#         item_i_per_relevance = item_per_relevance.unsqueeze(-1)
#         item_j_per_relevance = item_per_relevance.unsqueeze(-2)
#         paired_items = item_i_per_relevance + item_j_per_relevance
        
#         is_not_nan = ~torch.isnan(paired_items)
#         paired_items = paired_items[is_not_nan]
        
#         bpr = torch.log(torch.sigmoid(paired_items)+self.eps)
        
#         bpr = -bpr.mean()

#         return bpr