import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class Dataset_ETT(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 data_path='ETTh1.csv', scale=True):
        self.seq_len = size[0]
        self.pred_len = size[1]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Determina se il dataset è orario (h) o minuti (m) dal nome file
        # ETTh1/h2: 12 mesi train, 4 val, 4 test (1 punto all'ora)
        # ETTm1/m2: 12 mesi train, 4 val, 4 test (4 punti all'ora)
        
        if 'm' in self.data_path.lower():
            # 15 minuti = 4 punti/ora * 24 ore * 30 giorni = 2880 punti/mese
            steps_per_month = 4 * 24 * 30
        else:
            # Orario = 1 punto/ora * 24 ore * 30 giorni = 720 punti/mese
            steps_per_month = 1 * 24 * 30
            
        train_steps = 12 * steps_per_month
        val_steps = 4 * steps_per_month
        test_steps = 4 * steps_per_month
        
        border1s = [0, train_steps - self.seq_len, train_steps + val_steps - self.seq_len]
        border2s = [train_steps, train_steps + val_steps, train_steps + val_steps + test_steps]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Features
        df_data = df_raw.iloc[:, 1:]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_dataloader(root_path, data_path, flag, seq_len, pred_len, batch_size, num_workers=0):
    dataset = Dataset_ETT(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, pred_len],
        scale=True
    )
    
    shuffle_flag = True if flag == 'train' else False
    drop_last = True if flag == 'train' else False
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return dataset, loader