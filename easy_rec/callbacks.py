import torch
import pytorch_lightning as pl
class DynamicNegatives(pl.callbacks.Callback):
    def __init__(self, dataloader, neg_key = "out_sid", id_key = "uid", padding_idx = 0):
        super().__init__()
        self.dataloader = dataloader

        self.neg_key = neg_key
        self.id_key = id_key
        self.padding_idx = padding_idx

        self.dataloader.negatives_buffer = {}

        self.init_vars()
    
    def init_vars(self):
        self.id_keys = []
        self.sampled_negatives = []
        self.predictions_pos = []
        self.predictions_neg = []
        
    def on_train_batch_end(self, trainer, pl_module, model_outputs, batch_input, batch_idx):
        model_output = model_outputs['model_output']
        self.predictions_pos.append(model_output[:,:,:1])
        self.predictions_neg.append(model_output[:,:,1:])
        self.sampled_negatives.append(batch_input[self.neg_key][:,:,1:])
        self.id_keys.append(batch_input[self.id_key])

    def on_train_epoch_end(self, trainer, pl_module):
        # Reshape of buffer and predictions
        self.sampled_negatives = torch.cat(self.sampled_negatives)
        self.predictions_neg = torch.cat(self.predictions_neg)
        self.predictions_pos = torch.cat(self.predictions_pos)
        self.id_keys = torch.cat(self.id_keys)

        mask = self.predictions_neg >= self.predictions_pos # compare the negative scores with the target one

        negatives_buffer = {}
        for i, id_key in enumerate(self.id_keys):
            neg_set = set(self.sampled_negatives[i][mask[i]].flatten().tolist())
            neg_set = neg_set - {self.padding_idx}
            negatives_buffer[id_key.item()] = neg_set
        self.init_vars()

        self.dataloader.collate_fn.update_buffer(negatives_buffer)

# def update_buffer(self,new_negative_buffer):
#     self.negatives_buffer = new_negative_buffer.copy()

# def dynamic_negatives(self, possible_negatives, n, i):
#     # If the buffer is empty, sample uniformly
#     if len(self.negatives_buffer) == 0:
#         return self.uniform_negatives(possible_negatives, n)
    
#     # Get the negatives with score higher than the target in the previous epoch
#     new_negatives = torch.tensor(list(self.negatives_buffer[i]))

#     if len(new_negatives) < n:
#         new_negatives = torch.cat([new_negatives, self.uniform_negatives(possible_negatives, n-len(new_negatives))])
    
#     new_negatives = new_negatives[torch.randperm(len(new_negatives))] #shuffle new_negatives
#     #new_negatives = new_negatives.reshape(1, -1, self.num_negatives)

#     return new_negatives