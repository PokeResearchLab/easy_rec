import torch

class SequentialBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        is_not_nan = ~torch.isnan(target)

        output = super().forward(input[is_not_nan], target[is_not_nan])

        return output
    
class SequentialCrossEntropyLoss(torch.nn.Module): #torch.nn.CrossEntropyLoss):
    def __init__(self, eps = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, input, target):
        is_not_nan = ~torch.isnan(target)
        print("A")
        
        #Manual computation, cause CrossEntropyLoss returns nan
        new_target = target
        new_target[~is_not_nan] = 0
        print("B")
        
        exps = torch.exp(input) * is_not_nan
        exps_sum = exps.sum(dim=-1)
        print("exps_sum", exps_sum)

        exps_div = exps/(exps_sum.unsqueeze(-1)+self.eps)
        exps_div = exps_div * is_not_nan
        print("exps_div", exps_div)

        loss = exps_div*torch.log(exps_div)*new_target
        print("E")

        loss = -loss.sum(dim=-1)[is_not_nan.any(dim=-1)]
        print("F")

        output = loss.mean()
        print("output", output)

        # Commented code cause CrossEntropyLoss returns nan
        # target[is_nan] = 0
        # input[is_nan] = -100

        # all_items_nans = is_nan.all(dim=-1)

        # new_target = target[~all_items_nans]
        # new_input = input[~all_items_nans]

        # output = super().forward(input, target)

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
        positive_items = new_target > 0
        negative_items = new_target < 0
        item_pairs = (positive_items.unsqueeze(-1) * negative_items.unsqueeze(-2)).float()
        # pair positive and negative items in same timestep
        item_per_relevance = item_per_relevance.unsqueeze(-1) - item_per_relevance.unsqueeze(-2)
        item_per_relevance = -torch.log(1+torch.exp(-item_per_relevance))
        # item_per_relevance has shape (N,T,I,I)
        # item_pairs has shape (N,T,I,I)
        # We want shape (N,T,1). summing on last two dimensions if item_pairs is True
        bpr = torch.einsum('ntij,ntij->nt', item_per_relevance, item_pairs)

        bpr = bpr[is_not_nan.any(dim=-1)]
        
        bpr = -bpr.mean()

        return bpr