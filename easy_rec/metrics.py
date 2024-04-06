import torch
import torchmetrics

class RecMetric(torchmetrics.Metric):
    """
        Initializes the RecMetric.

        Args:
            top_k (list): List of integers representing top-k values for evaluation.
            batch_metric (bool): Whether to compute metrics on batch level or not.
    """
    def __init__(self, top_k=[5,10,20], batch_metric = False):
        super().__init__()
        self.top_k = top_k if isinstance(top_k, list) else [top_k]
        self.batch_metric = batch_metric

        # Initialize state variables for correct predictions and total examples
        for top_k in self.top_k:
            if not self.batch_metric:
                self.add_state(f"correct@{top_k}", default=torch.tensor(0.), dist_reduce_fx="sum")
            else:
                self.add_state(f"correct@{top_k}", default=[], dist_reduce_fx="cat")
        
        if not self.batch_metric:
            self.add_state(f"total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self):
        """
        Computes and returns the metric values.

        Returns:
            dict: Dictionary containing metric values.
        """
        # Compute accuracy as the ratio of correct predictions to total examples
        out = {}
        for k in self.top_k:
            out[f"@{k}"] = getattr(self, f"correct@{k}") 
            if not self.batch_metric:
                out[f"@{k}"] = out[f"@{k}"] / self.total
            else:
                out[f"@{k}"] = torchmetrics.utilities.dim_zero_cat(out[f"@{k}"])
        return out
    
    def not_nan_subset(self, **kwargs):
        """
        Subsets input tensors where the 'relevance' tensor is not NaN.

        Returns:
            dict: Subset of input tensors where 'relevance' is not NaN.
        """
        if "relevance" in kwargs:
            # Subset other args, kwargs where relevance is not nan
            relevance = kwargs["relevance"]
            is_not_nan_per_sample = ~torch.isnan(relevance).any(-1)
            kwargs = {k: v[is_not_nan_per_sample] for k, v in kwargs.items()}
            # This keeps just the last dimension, the others are collapsed

        return kwargs
    
class NDCG(RecMetric):
    '''
     Normalized Discounted Cumulative Gain (NDCG) assesses the performance of a ranking system by considering the placement of K relevant items 
     within the ranked list. The underlying principle is that items higher in the ranking should receive a higher score than those positioned 
     lower in the list because they are those where a user's attention is usually focused.
    '''
    def __init__(self, top_k=[5,10,20], batch_metric=False):
        super().__init__(top_k, batch_metric)
        self.batch_metric=batch_metric

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        """
        Updates the metric values based on the input scores and relevance tensors.

        Args:
            scores (torch.Tensor): Tensor containing prediction scores.
            relevance (torch.Tensor): Tensor containing relevance values.

     """
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        app = torch.log2(ranks+1)
        for top_k in self.top_k:
            dcg = ((ranks<=top_k)*relevance/app).sum(-1)
            k = min(top_k,scores.shape[-1])
            sorted_k_relevance = relevance.sort(dim=-1, descending=True).values[...,:k] #get first k items in sorted_relevance on last dimension  
            idcg = (sorted_k_relevance/torch.log2(torch.arange(1,k+1,device=sorted_k_relevance.device)+1)).sum(-1)
            ndcg = dcg/idcg # ndcg.shape = (num_samples, lookback)
            if not self.batch_metric:
                setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + ndcg.sum())
            else:
                getattr(self, f"correct@{top_k}").append(ndcg)
        if not self.batch_metric:
            self.total += relevance.shape[0]
    
class MRR(RecMetric):
    '''
    Mean Reciprocal Rank (MRR) evaluates the efficacy of a ranking system by considering the placement of the first relevant item within the ranked list.
    It is calculated by taking the reciprocal of the rank of the first relevant item.
    It emphasizes that the position of the first relevant item is more important than the placement of the other relevant items.
    '''
    def __init__(self, top_k=[5,10,20], batch_metric=False):
        super().__init__(top_k, batch_metric)
        self.batch_metric = batch_metric

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for top_k in self.top_k:
            mrr = ((ranks<=top_k)*relevant*(1/ranks)).max(-1).values
            if not self.batch_metric:
                setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + mrr.sum())
            else:
                getattr(self, f"correct@{top_k}").append(mrr)
        if not self.batch_metric:
            self.total += relevance.shape[0]

class Precision(RecMetric):
    '''
    It computes the proportion of accurately identified relevant items among all the items recommended within a list of length K. 
    It is used to explicitly count the number of recommended, or retrieved, items that are truly relevant.
    '''
    def __init__(self, top_k=[5,10,20], batch_metric=False):
        super().__init__(top_k, batch_metric)
        self.batch_metric = batch_metric

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        """
        Updates the metric values based on the input scores and relevance tensors.

        Args:
            scores (torch.Tensor): Tensor containing prediction scores.
            relevance (torch.Tensor): Tensor containing relevance values.
        """
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for top_k in self.top_k:
            precision = ((ranks<=top_k)*relevant/top_k).sum(-1)
            if not self.batch_metric:
                setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + precision.sum())
            else:
                getattr(self, f"correct@{top_k}").append(precision)
        if not self.batch_metric:
            self.total += relevance.shape[0]

class Recall(RecMetric):
    '''
    It assesses the fraction of correctly identified relevant items among the top K recommendations, relative to the total number of relevant items in the dataset. 
    It measures the effectiveness of the method in capturing relevant items among all of those present in the dataset.
    
    '''
    def __init__(self, top_k=[5,10,20], batch_metric=False):
        super().__init__(top_k, batch_metric)
        self.batch_metric=batch_metric

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for top_k in self.top_k:
            recall = ((ranks<=top_k)*relevant/relevant.sum(-1,keepdim=True)).sum(-1)
            #torch.minimum(relevant.sum(-1,keepdim=True),top_k*torch.ones_like(relevant.sum(-1,keepdim=True)))
            if not self.batch_metric:
                setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + recall.sum())
            else:
                getattr(self, f"correct@{top_k}").append(recall)
        if not self.batch_metric:
            self.total += relevance.shape[0]

class F1(RecMetric):
    '''
    The F1 score is the harmonic mean of precision and recall. 
    It is a single metric that combines both precision and recall to provide a single measure of the quality of a ranking system.
    '''
    def __init__(self, top_k=[5,10,20], batch_metric=False):
        super().__init__(top_k, batch_metric)
        self.precision = Precision(top_k, batch_metric)
        self.recall = Recall(top_k, batch_metric)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        self.precision.update(scores, relevance)
        self.recall.update(scores, relevance)

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        out = {}
        for k in self.top_k:
            out[f"@{k}"] = 2*(precision[f"@{k}"]*recall[f"@{k}"])/(precision[f"@{k}"]+recall[f"@{k}"])
        return out

class PrecisionWithRelevance(RecMetric): #TODO CHE ROBA è?
    '''
    It computes the proportion of accurately identified relevant items among all the items recommended within a list of length K.
    It is used to explicitly count the number of recommended, or retrieved, items that are truly relevant.
    '''
    def __init__(self, top_k=[5,10,20], batch_metric=False):
        super().__init__(top_k, batch_metric)
        self.batch_metric=batch_metric

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        for top_k in self.top_k:
            precision = ((ranks<=top_k)*relevance/(top_k*relevance.sum(-1,keepdim=True))).sum(-1)
            if not self.batch_metric:
                setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + precision.sum())
            else:
                getattr(self, f"correct@{top_k}").append(precision)

        if not self.batch_metric:
            self.total += relevance.shape[0]

class MAP(RecMetric):
    '''
    Mean Average Precision (MAP) evaluates the efficacy of a ranking system by considering the average precision across the top R recommendations for R ranging from 1 to K. 
    It emphasizes that precision values for items within the top K positions contribute to the overall assessment also accounting for the significance of the order in the ranking. 
    Different from NDCG, this metric does not explicitly assign a different importance to different slots.
    
    '''
    def __init__(self, top_k=[5,10,20], batch_metric=False):
        super().__init__(top_k, batch_metric)
        self.batch_metric=batch_metric

        self.precision_at_k = PrecisionWithRelevance(list(range(1,torch.max(torch.tensor(self.top_k))+1)), batch_metric)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        self.precision_at_k.update(scores, relevance)

    def compute(self):
        if not self.batch_metric:
            for top_k in self.top_k:
                for k in range(1,top_k+1):
                    setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + getattr(self.precision_at_k, f"correct@{k}"))
                setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}")/top_k)
            setattr(self,"total", getattr(self.precision_at_k, f"total"))
        else:
            for top_k in self.top_k:
                correct = getattr(self.precision_at_k, f"correct@{top_k}")
                for k in range(1,top_k):
                    new_correct = getattr(self.precision_at_k, f"correct@{k}")
                    for i,c in enumerate(new_correct):
                        correct[i] += c
                setattr(self, f"correct@{top_k}", [c/top_k for c in correct])
        
        return super().compute()
    
    def reset(self):
        super().reset()
        self.precision_at_k.reset()

