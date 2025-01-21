import torch
class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, emb_size, num_layers, graph, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        #self.keep_prob = self.config['keep_prob']
        #self.A_split = self.config['A_split']
        self.user_emb = torch.nn.Embedding(num_users+1, emb_size)
        self.item_emb = torch.nn.Embedding(num_items+1, emb_size)
        #self.graph but make it buffer so is moved to the GPU
        self.register_buffer('graph', graph)
        #make the graph non trainable
        self.graph.requires_grad = False
    
    def computer(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [all_emb]
        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.user_emb.weight.shape[0], self.item_emb.weight.shape[0]])
        return users, items
       
    def forward(self, users, items):
        items = items.squeeze()
        all_users, all_items = self.computer()
        users_emb = all_users[users].unsqueeze(1).repeat(1, items.shape[1], 1)
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        logits = torch.sum(inner_pro, dim=-1).unsqueeze(1)
        return logits