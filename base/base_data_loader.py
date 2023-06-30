import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd




# Dataloader that trains using all patient image
class BaseDataLoader(DataLoader):
    """
    Use all pat image
    """
    def __init__(self, dataset, batch_size, subject_map, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.subject_map = subject_map
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            #'subject_map': self.subject_map,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
        
        df = self.subject_map.copy()
        df["subject_no_full"] = df["stock"].astype(str) + "_" + df["date"].astype(str) + "_" + df['idx_num'].astype(str)
        stocks = df["stock"].unique()
        subjects = df["subject_no_full"].unique()
        
        # if isinstance(split, int):
        #     assert split > 0
        #     assert split < len(lst), "validation set size is configured to be larger than entire dataset."
        #     len_valid = split
        # else:
        #     len_valid = int(len(lst) * split)
        
        np.random.seed(0)        
        
        
        df_train = pd.DataFrame(columns=df.columns)
        df_valid = pd.DataFrame(columns=df.columns)

        for stock in stocks:
            df_tmp = df.loc[df["stock"]==stock]
            lst_tmp = df_tmp['subject_no_full'].unique()
            np.random.shuffle(lst_tmp)
            len_valid_tmp = int(df_tmp.shape[0]*split)
            valid_no_tmp = lst_tmp[0:len_valid_tmp]
            train_no_tmp = np.delete(lst_tmp, np.arange(0,len_valid_tmp))
            df_train_tmp= df_tmp.loc[df_tmp["subject_no_full"].isin(train_no_tmp)]
            df_valid_tmp= df_tmp.loc[df_tmp["subject_no_full"].isin(valid_no_tmp)]
            df_train=pd.concat([df_train,df_train_tmp])
            df_valid=pd.concat([df_valid,df_valid_tmp])

        print(df_train)
        print()
        print(df_valid)
        
        # valid_no = lst[0:len_valid]
        # train_no = np.delete(lst, np.arange(0, len_valid))
        
        
        # df_train = df.loc[df["subject_no_full"].isin(train_no)]
        # df_valid = df.loc[df["subject_no_full"].isin(valid_no)]
        
        
        print(len(np.array(df_train.index)))
        print(len(np.array(df_valid.index)))

        train_sampler = SubsetRandomSampler(np.array(df_train.index))
        valid_sampler = SubsetRandomSampler(np.array(df_valid.index))

        # turn off shuffle option which is mutually exclusive with sampler
        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class BaseDataLoader_basic(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, subject_map, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.subject_map = subject_map
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            #'subject_map': self.subject_map,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)
        
        
        # label_type_dict = {
        #     'BCC': 1,
        #     'SCC': 2,
        #     'Melanoma': 3,
        #     'Benign': 0
        # }
        
        # df_train = pd.DataFrame()
        # df_valid = pd.DataFrame()

        # for label in label_type_dict.keys():
        #     df_label = df_original.loc[df_original['cell_type']==label]
        #     patients = list(df_label['subject_no'].unique())
        #     len_train = int(len(patients)*0.9)
        #     patients_train = patients[:len_train]
        #     patients_valid = patients[len_train:]
        #     train = df_label.loc[df_label['subject_no'].isin(patients_train)]
        #     valid = df_label.loc[df_label['subject_no'].isin(patients_valid)]

        #     df_train = df_train.append(train)
        #     df_valid = df_valid.append(valid)  
        
        

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)



