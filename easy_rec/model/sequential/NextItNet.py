import torch
import torch.nn as nn
class NextItNet(nn.Module):
    def __init__(self, num_users, num_items, emb_size, dilation_rates, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.user_emb = nn.Embedding(num_users+1, emb_size)
        self.item_emb = nn.Embedding(num_items+1, emb_size)
        self.blocks = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        for rate in dilation_rates:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(emb_size, emb_size, kernel_size=3, padding=rate, dilation=rate),
                    nn.ReLU(),
                    nn.BatchNorm1d(emb_size),
                    nn.Dropout(dropout_rate),
                )
            )
        self.fc_out = nn.Linear(emb_size, num_items+1)

    def forward(self, item_seq, items_to_predict, user_ids):
        
        # Get embeddings
        user_emb = self.user_emb(user_ids)  # (batch, emb_size)
        item_emb = self.item_emb(item_seq)  # (batch, sequence_length, emb_size)

        # Combine user and item embeddings
        user_emb = user_emb.unsqueeze(1)  # (batch, 1, emb_size)
        x = user_emb * item_emb  # (batch, sequence_length, emb_size)

        # Rearrange for Conv1d: (batch, emb_size, sequence_length)
        x = x.permute(0, 2, 1)

        # Apply dropout
        x = self.dropout(x)

        # Pass through dilated convolutions
        for block in self.blocks:
            x = block(x)

        #x = x[:, :, -1]  # (batch, emb_size)
        x = x.permute(0, 2, 1) #(batch, sequence_length, emb_size)

        scores = self.fc_out(x)#.unsqueeze(1)  # (batch, sequence_length, num_items+1)
        timesteps_to_use = min(items_to_predict.shape[1], scores.shape[1])
        #print(timesteps_to_use.shape)

        scores = scores[:, -timesteps_to_use:]
        items_to_predict = items_to_predict[:, -timesteps_to_use:]

        scores = torch.gather(scores, -1, items_to_predict) # Get scores for items in items_to_predict
        return scores