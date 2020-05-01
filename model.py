class EmbeddingLayer(nn.Module):
    def __init__(self, unique_vals):
        super(EmbeddingLayer, self).__init__()
        
        self.embed_list = [nn.Embedding(100000, int(min(np.ceil(val / 2), 64))) for val in unique_vals]
            
    def forward(self, inputs):
        emb = [emb(torch.tensor((inputs[:, idx]), dtype=torch.long)) for idx, emb in zip(list([i for i in range(23)]), self.embed_list)]
        out = torch.cat([e for e in emb], dim=1)
        return out


class Model(nn.Module):
    def __init__(self, input_size, unique_vals, device, hidden_size=1024, dropout=0.5):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.embedding = EmbeddingLayer(unique_vals)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, le_df):
        embed = self.embedding(le_df)
        embed = embed.to(self.device)
        out = self.bn1(self.relu(self.fc1(embed)))
        
        out = self.dropout(out)
        
        out = self.bn2(self.relu(self.fc2(out)))
        out = self.dropout(out)
        
        out = self.fc3(out)
        #print(out.size(), '------')
        return self.sigmoid(out)
 

model = Model(vector_size, unique_vals, device).to(device)