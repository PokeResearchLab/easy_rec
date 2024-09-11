import torch

class NCF(torch.nn.Module):
    def __init__(self, num_users, num_items, mf_emb_size, mlp_emb_size, layers, **kwargs):
        super().__init__()
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=num_users+1, embedding_dim=mlp_emb_size)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=num_items+1, embedding_dim=mlp_emb_size)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=num_users+1, embedding_dim=mf_emb_size)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=num_items+1, embedding_dim=mf_emb_size)

        self.fc_layers = torch.nn.ModuleList()
        layers_sizes = [2*mlp_emb_size, *layers]
        for idx, (in_size, out_size) in enumerate(zip(layers_sizes[:-1], layers_sizes[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers_sizes[-1] + mf_emb_size, out_features=1)

    def forward(self, user_indices, item_indices):
        item_indices = item_indices.squeeze()
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        #Repeat user_embedding_mlp to match the shape of item_embedding_mlp
        #user_embedding_mlp from shape (batch, mlp_emb_size) to (batch, item_num, mlp_emb_size)
        user_embedding_mlp = user_embedding_mlp.unsqueeze(1).repeat(1, item_indices.shape[1], 1)
        #Do the same for user_embedding_mf
        user_embedding_mf = user_embedding_mf.unsqueeze(1).repeat(1, item_indices.shape[1], 1)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)

        logits = logits.squeeze().unsqueeze(1)

        return logits
