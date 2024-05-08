# import torch

# class Seq2Rec(torch.nn.Module):
#     def __init__(self, 
#                  embedding_layer,
#                  encoder_layer,
#                  output_layer, 
#                  **kwargs):
#         '''
#         TODO
#         '''
#         super().__init__()


#     def forward(self, input_sequences, output_items):
#         '''
#         TODO
#         '''
        
#         embedded_input = self.embedding_layer(input_sequences)

#         encoded = self.encoder_layer(embedded_input)

#         output_scores = self.output_layer(encoded, output_items)

        

#         return scores
    

# class ItemEmbedding(torch.nn.Module):
#     def __init__(self, num_items, emb_size):
#         super().__init__()
#         self.item_emb = torch.nn.Embedding(num_items, emb_size)

#     def forward(self, input_seqs):
#         return self.item_emb(input_seqs)
    
# class PositionalEmbedding(torch.nn.Module):
#     def __init__(self, lookback, emb_size):
#         super().__init__()
#         self.pos_emb = torch.nn.Embedding(lookback, emb_size)

#     def forward(self, input_seqs):
#         positions = torch.tile(torch.arange(input_seqs.shape[1], device=next(self.parameters()).device), [input_seqs.shape[0], 1])
#         return self.pos_emb(positions)
    
# class ItemPositionalEmbedding(torch.nn.Module):
#     def __init__(self, num_items, lookback, emb_size, aggregation='add'):
#         super().__init__()
#         self.item_emb = ItemEmbedding(num_items, emb_size)
#         self.pos_emb = PositionalEmbedding(lookback, emb_size)
#         if aggregation == 'concat':
#             self.aggregation = torch.cat
#         elif aggregation == 'add':
#             self.aggregation = torch.add
#         else:
#             raise ValueError('Invalid aggregation method')
        

#     def forward(self, input_seqs):
#         item

#         embedded = self.dropout(self.first_layernorm(self.item_emb(input_seqs) + self.pos_emb(positions)))

#         attention_mask = ~torch.tril(torch.ones((input_seqs.shape[1], input_seqs.shape[1]), dtype=torch.bool, device=next(self.parameters()).device))

#         encoded = self.encoder(embedded, attention_mask)

#         log_feats = encoded.unsqueeze(2)

#         poss_item_embs = self.item_emb(poss_item_seqs)

#         # Use only last timesteps in poss_item_seqs --> cut log_feats to match poss_item_embs
#         log_feats = log_feats[:, -poss_item_embs.shape[1]:, :, :] # (B, T, 1, E)

#         scores = (log_feats * poss_item_embs).sum(dim=-1)