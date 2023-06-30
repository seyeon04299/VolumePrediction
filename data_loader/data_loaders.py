from torchvision import datasets, transforms
from base import BaseDataLoader
from base import BaseDataLoader_basic
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from .datasets import *
import numpy as np



class DataLoader_NASDAQv_Final(BaseDataLoader):
    """
    NASDAQ VOLUME DATA LOADING using subject maps, for data from 2021.09.01 to 2022.03.31
    """
    def __init__(self, data_dir, batch_size, subject_map, start_ind, label_len, seq_len, pred_len, sec, freq, lob=True, peak=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'


        self.data_dir = data_dir
        
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = Dataset_NASDAQv_Final(self.subject_map, mode, start_ind, label_len,seq_len, pred_len, sec, freq, lob, peak)
        self.subject_map = self.dataset.subject_map
        self.batch_size = batch_size
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)


class DataLoader_NASDAQv_Wholedata(BaseDataLoader):
    """
    NASDAQ VOLUME DATA LOADING using subject maps - for transformers
    """
    def __init__(self, data_dir, batch_size, subject_map, start_ind, label_len, seq_len, pred_len, sec, freq, lob=True, peak=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'


        self.data_dir = data_dir
        
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = Dataset_NASDAQv_Wholedata(self.subject_map, mode, start_ind, label_len,seq_len, pred_len, sec, freq, lob, peak)
        self.subject_map = self.dataset.subject_map
        self.batch_size = batch_size
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)



class DataLoader_NASDAQv_Transformers(BaseDataLoader):
    """
    NASDAQ VOLUME DATA LOADING using subject maps - for transformers
    """
    def __init__(self, data_dir, batch_size, subject_map, start_ind, label_len, seq_len, pred_len, sec, freq, lob=True, peak=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'


        self.data_dir = data_dir
        
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = Dataset_NASDAQv_Transformers(self.subject_map, mode, start_ind, label_len,seq_len, pred_len, sec, freq, lob, peak)
        self.subject_map = self.dataset.subject_map
        self.batch_size = batch_size
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)





class DataLoader_NASDAQv_AST(BaseDataLoader):
    """
    NASDAQ VOLUME DATA LOADING using subject maps (Used for AST)
    """
    def __init__(self, data_dir, batch_size, subject_map, start_ind,end_ind,p,q, sec, covariates=True,lob=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'


        self.data_dir = data_dir
        
        
        self.dataset = Dataset_NASDAQv_AST(self.subject_map, mode, start_ind, end_ind, p, q, sec, lob,covariates)
        self.subject_map = self.dataset.subject_map
        self.batch_size = batch_size
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)







class DataLoader_VolumeAlloc(BaseDataLoader):
    """
    NASDAQ VOLUME data loading for GAN
    """
    def __init__(self, data_dir, batch_size, subject_map, start_ind,end_ind,p,q, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'


        self.data_dir = data_dir
        
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = Dataset_TimeGAN(self.subject_map, mode, start_ind, end_ind, p, q)
        self.subject_map = self.dataset.subject_map
        self.batch_size = batch_size
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)








