class Dataset(object):
    def __init__(self, df, features):
        self.df = df.drop(['id', 'target'], axis=1).values
        self.target = df.target.values
        self.features = features
        self.unique_values = [int(df[feat].nunique()) for feat in self.features]
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        inputs = self.df[idx]
        targets = self.target[idx]
        unique_vals = self.unique_values
        sample = {'inputs': inputs, 'targets': targets, 'unique': unique_vals}
        
        return sample