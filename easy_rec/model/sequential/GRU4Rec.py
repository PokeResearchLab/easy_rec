import torch

class GRU4Rec(torch.nn.Module):
    def __init__(self, 
                 num_items,
                 emb_size,
                 num_layers=1,
                 dropout_hidden=0, 
                 dropout_input=0,
                 padding_value = 0,
                 **kwargs):
        '''
        args:
            num_items (int): Number of items in the dataset.
            hidden_size (int): Size of the hidden state in the GRU.
            num_layers (int, optional): Number of layers in the GRU. Defaults to 1.
            dropout_hidden (float, optional): Dropout rate for hidden states in the GRU. Defaults to 0.
            dropout_input (float, optional): Dropout rate for input embeddings. Defaults to 0.
            emb_size (int, optional): Size of the item embedding. Defaults to 50.
        '''
        
        super(GRU4Rec, self).__init__()

        # Initialize model parameters
        self.num_items = num_items
        
        hidden = torch.zeros(num_layers, emb_size, requires_grad=True)
        self.register_buffer("hidden", hidden) #register buffer is needed to move the tensor to the right device

        # Dropout layer for input embeddings
        self.inp_dropout = torch.nn.Dropout(p=dropout_input)

        # Linear layer for output logits
        #self.h2o = torch.nn.Linear(hidden_size, num_items+1)

        # Item embedding layer
        self.item_emb = torch.nn.Embedding(num_items+1, emb_size, padding_idx=padding_value)
        
        # GRU layer
        self.gru = torch.nn.GRU(emb_size, emb_size, num_layers, dropout=dropout_hidden, batch_first=True)

    def forward(self, input_seqs, items_to_predict):
        ''' 
        Input:
            input_seqs (torch.Tensor): Tensor containing input item sequences. Shape (batch_size, sequence_length).
            items_to_predict (torch.Tensor): Tensor containing possible item sequences. Shape (batch_size, input_seq_len, output_seq_len, num_items)

        Output:
            scores (torch.Tensor): Tensor containing interaction scores between input and possible items. Shape (batch_size, input_seq_len, output_seq_len, num_items)
        '''

        embedded = self.item_emb(input_seqs)

        embedded = self.inp_dropout(embedded)

        hidden_repeated = self.hidden.unsqueeze(1).repeat(1, input_seqs.shape[0], 1) #torch.tile(self.hidden, (1, input_seqs.shape[0], 1))

        output, hidden = self.gru(embedded, hidden_repeated)

        encoded = output.unsqueeze(2)
        poss_item_embs = self.item_emb(items_to_predict)
        # Use only last timesteps in items_to_predict --> cut log_feats to match poss_item_embs
        encoded = encoded[:, -poss_item_embs.shape[1]:, :, :] # (B, T, 1, E)
        scores = (encoded * poss_item_embs).sum(dim=-1)

        # scores = self.h2o(output)
        # scores = scores[:, -items_to_predict.shape[1]:, :]
        # scores = torch.gather(scores, -1, items_to_predict) # Get scores for items in items_to_predict

        return scores