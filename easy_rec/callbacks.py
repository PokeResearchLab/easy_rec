import torch
import pytorch_lightning as pl
class DynamicNegatives(pl.callbacks.Callback):
    def __init__(self, dataloader, neg_key = "out_sid", id_key = "uid"):
        super().__init__()
        self.dataloader = dataloader

        self.neg_key = neg_key
        self.id_key = id_key

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
        sampled_negatives_reshaped = self.sampled_negatives.reshape(self.sampled_negatives.shape[0], -1)
        predictions_pos_reshaped = self.predictions_pos.reshape(self.predictions_pos.shape[0], -1)
        predictions_neg_reshaped = self.predictions_neg.reshape(self.predictions_neg.shape[0], -1)

        mask = predictions_neg_reshaped >= predictions_pos_reshaped # compare the negative scores with the target one

        negatives_buffer = {}
        for i, id_key in enumerate(self.id_keys):
            negatives_buffer[id_key] = sampled_negatives_reshaped[i][mask[i]]
        self.init_vars()

        self.dataloader.collate_fn.update_buffer(negatives_buffer)


#loader_params = cfg["model"]["loader_params"].copy()
#loader_params['multiprocessing_context']='fork' if torch.backends.mps.is_available() else None