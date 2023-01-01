from torch.utils.data import Dataset, DataLoader , Subset
import numpy as np
import torch as tc



class GeneralDataset(Dataset):
    def __init__(self, data, seq_len,batch_size,bpe):
        super().__init__()
        self._data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.bpe = bpe

    @property
    def data(self):
        return self._data.copy()

    def __len__(self):
        return len(self._data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self._data[idx:idx + self.seq_len]
        y = self._data[idx + 1:idx + self.seq_len + 1]

        inp = tc.FloatTensor(x.reshape(self.seq_len, -1))
        target = tc.FloatTensor(y.reshape(self.seq_len, -1))

        return inp, target



    def get_rand_dataloader(self):
        indices = np.random.permutation(len(self))[:self.bpe*self.batch_size]
        subset = Subset(self, indices.tolist())
        dataloader = DataLoader(subset, batch_size=self.batch_size)
        return dataloader