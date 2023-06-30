
from cProfile import label
from torch import float32
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from scipy import stats
from data_loader.src.constants import REGMKT_END_TIME_NS, REGMKT_START_TIME_NS, INTERVAL_NS, R_N,A_N
from data_loader.src.timefeatures import time_features

from utils import normalize_max, normalize_historic, peak_identification



# _min_to_ns = lambda x: int(x * 60 * 1e9)
# _sec_to_ns = lambda x: int(x * 1e9)

# # 9:30 (i.e. regular market start time) in nanoseconds
# REGMKT_START_TIME_MIN = 9 * 60 + 30 # 9:30am
# REGMKT_START_TIME_NS = _min_to_ns(REGMKT_START_TIME_MIN)

# # 16:00 in nanoseconds
# REGMKT_END_TIME_MIN = 16 * 60 # 16:00 or 4pm
# REGMKT_END_TIME_NS = _min_to_ns(REGMKT_END_TIME_MIN)
# # interval in nanoseconds
# INTERVAL_SEC = 1
# INTERVAL_NS = _sec_to_ns(INTERVAL_SEC)
REG_N = (REGMKT_END_TIME_NS - REGMKT_START_TIME_NS -1) // INTERVAL_NS +1





class Dataset_NASDAQv_Final(Dataset): 
    '''
    Dataset for Callling Volume & LOB data (2021.09.01~2022.03.31)
    '''

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map, mode, start_ind, label_len, seq_len, pred_len, sec=1,freq='t',lob=True,peak=False):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        ## used for indexing the securities - unique number for securities
        self.securities = pd.read_csv('./data/NASDAQ_used/final_stocks_nasdaq.csv')

        ## Precalculated volume averages used for normalization
        self.avgs = pd.read_csv('./data/NASDAQ_used/historical_stats_2109_2203.csv')
        self.avgs.rename(columns = {'vol_mean':'vol1_mean', 'vol_std':'vol1_std'}, inplace = True)
        
        # use_stocks_tmp = pd.read_csv('./data/NASDAQ_used/nasdaq_screener_1652753301137.csv')
        # self.use_stocks = list(use_stocks_tmp['Symbol'])

        use_stocks_tmp = pd.read_csv('./data/NASDAQ_used/high_volume_stocks.csv')        # High volume stocks
        self.use_stocks = list(use_stocks_tmp['Symbols'])

        # self.use_stocks = ['AAPL'] #,'MSFT','GOOG','GOOGL','FB','NVDA','TSLA','INTC','AVGO','CSCO','COST','ASML','PEP','ADBE','TXN','QCOM','INTU','ADI','MU','LRCX','FISV','ATVI']

        self.subject_map = self.subject_map[self.subject_map['stock'].isin(self.use_stocks)].reset_index(drop=True)

        self.mode = mode

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        initial_len = 23400     ## 60*60*6.5 seconds in a whole day (9.30~16.00)

        self.start_ind = start_ind
        # self.end_ind = start_ind+self.pred_len + self.seq_len        
        self.sec = sec
        assert initial_len%self.sec==0, 'Remainder of (23400 / sec) must be 0'
        ## duplicate rows to use multiple timeframes from single stock
        shapes = self.subject_map.shape
        
        self.whole_len = self.seq_len+self.pred_len
        initial_len_new = int(initial_len/self.sec)
        nums_data_per_row = int(initial_len_new/self.whole_len)
        seq_idx = np.array(list(np.arange(nums_data_per_row))*shapes[0])

        self.subject_map = self.subject_map.loc[self.subject_map.index.repeat(nums_data_per_row)].reset_index(drop=True)
        self.subject_map['idx_num']=seq_idx
        

        self.peak = peak
        # Randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index(drop=True)

        # Time Features
        self.freq = freq
        # self.num_covariates = 1
        
        if self.freq=='t':          ## Currently onyl implement for seconds 't'
            self.cov = np.linspace(REGMKT_START_TIME_NS+60*10**9,REGMKT_END_TIME_NS,int(REG_N/self.sec))
            self.cov = stats.zscore(self.cov)
            self.num_covariates=2

        self.lob = lob


        ### If only the observed time want to be used instead of the whole day
        # self.subject_map = self.subject_map[self.subject_map['idx_num']==0].reset_index(drop=True)


        print("Dataset")
        self.len = len(self.subject_map)

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수 
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        stock = row['stock']
        ### Hitorical Mean & Stds for normalizations
        historical = self.avgs[self.avgs['security']==stock].reset_index(drop=True)
        mean = historical['vol'+str(self.sec)+'_mean'][0]
        std = historical['vol'+str(self.sec)+'_std'][0]

        stat1 = (historical['vol1_mean'][0], historical['vol1_std'][0])
        stat2 = (historical['vol2_mean'][0], historical['vol2_std'][0])
        stat5 = (historical['vol5_mean'][0], historical['vol5_std'][0])
        stat10 = (historical['vol10_mean'][0], historical['vol10_std'][0])
        stat30 = (historical['vol30_mean'][0], historical['vol30_std'][0])
        stat60 = (historical['vol60_mean'][0], historical['vol60_std'][0])
        

        lobv_mean = historical['lob_v_mean'][0]
        lobv_std = historical['lob_v_std'][0]
        lobp_mean = historical['lob_p_mean'][0]
        lobp_std = historical['lob_p_std'][0]
        
        ### Security idx for future use (embedding security information)
        idx_security = self.securities[self.securities['Symbol']==row['stock']].index[0]

        ### Volume Sequence idx
        idx_num = row['idx_num']
        start_ind = idx_num*self.whole_len
        end_ind = (idx_num+1)*self.whole_len

        ### Load and Normalize Volume Data
        v_data = np.load(row['vol_dir'])['sequence'].astype(float)
        if self.sec!=1:
            v_data = np.sum(v_data.reshape(int(v_data.shape[0]/self.sec),-1),axis=1)
        v_data = normalize_historic(v_data,mean,std)
        v_data = v_data[start_ind:end_ind]
        v_data = v_data.reshape(-1,1)

        ### Load and Normalize LOB data
        if self.lob:
            lob = np.load(row['lob_dir'])['sequence'].astype(float)
            lob[:,0::2] = normalize_historic(lob[:,0::2],lobv_mean,lobv_std)
            lob[:,1::2] = normalize_historic(lob[:,0::2],lobp_mean,lobp_std)

            lob = lob[self.sec-1::self.sec]
            lob = lob[start_ind:end_ind]

            self.num_covariates=22 if self.peak else 21

        ### Preallocation of final data
        # data = np.zeros((v_data.shape[0],self.num_covariates))
        # data[:,0]=v_data

        if self.peak:
            
            # peaks = peak_identification(v_data,4,6,0.8)
            peaks1 = np.where(v_data>2,0.5,0)
            peaks2 = np.where(v_data>3,0.5,0)
            peaks = peaks1+peaks2
            # peaks = np.expand_dims(peaks["signals"], axis=1)
            data = np.concatenate([v_data,peaks], axis=1)

            if self.lob:
                data = np.concatenate([v_data,peaks,lob], axis=1)

        else:
            if self.lob:
                data = np.concatenate([v_data,lob], axis=1)

        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len



        data_x = data[s_begin:s_end,:]
        data_y = data[r_begin:r_end,:]
        
        
        cov=self.cov[start_ind:end_ind]
        
        data_x_mark = np.expand_dims(cov[s_begin:s_end], axis=-1)
        data_y_mark = np.expand_dims(cov[r_begin:r_end], axis=-1)

        # print(data.shape)
        
        # Xmb = normalize_max(row,self.p)
        return (data_x, data_y, data_x_mark, data_y_mark, stat1,stat2,stat5,stat10,stat30,stat60)







class Dataset_NASDAQv_Wholedata(Dataset): 
    '''
    Used for Autoformer, Informer, Reformer, ...
    '''

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map, mode, start_ind, label_len, seq_len, pred_len, sec=1,freq='t',lob=True,peak=False):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        ## used for indexing the securities - unique number for securities
        self.securities = pd.read_csv('./data/NASDAQ_used/final_stocks_nasdaq.csv')

        ## Precalculated volume averages used for normalization
        self.avgs = pd.read_csv('./data/NASDAQ_used/volume_avg.csv')
        self.avgs.rename(columns = {'mean_vol':'mean_vol1', 'std_vol':'std_vol1'}, inplace = True)
        
        # use_stocks_tmp = pd.read_csv('./data/NASDAQ/nasdaq_screener_1652753301137.csv')
        # self.use_stocks = list(use_stocks_tmp['Symbol'])

        use_stocks_tmp = pd.read_csv('./data/NASDAQ_used/high_volume_stocks.csv')        # High volume stocks
        self.use_stocks = list(use_stocks_tmp['Symbols'])

        # self.use_stocks = ['AMZN','MSFT','GOOG','GOOGL','FB','NVDA','TSLA','INTC','AVGO','CSCO','COST','ASML','PEP','ADBE','TXN','QCOM','INTU','ADI','MU','LRCX','FISV','ATVI']

        self.subject_map = self.subject_map[self.subject_map['stock'].isin(self.use_stocks)].reset_index()

        self.mode = mode

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        initial_len = 23400     ## 60*60*6.5 seconds in a whole day (9.30~16.00)

        # self.start_ind = start_ind
        # self.end_ind = start_ind+self.pred_len + self.seq_len        
        self.sec = sec
        assert initial_len%self.sec==0, 'Remainder of (23400 / sec) must be 0'
        ## duplicate rows to use multiple timeframes from single stock
        shapes = self.subject_map.shape
        
        self.whole_len = self.seq_len+self.pred_len
        initial_len_new = int(initial_len/self.sec)
        nums_data_per_row = int(initial_len_new/self.whole_len)
        seq_idx = np.array(list(np.arange(nums_data_per_row))*shapes[0])

        self.subject_map = self.subject_map.loc[self.subject_map.index.repeat(nums_data_per_row)].reset_index(drop=True)
        self.subject_map['idx_num']=seq_idx
        

        self.peak = peak
        # Randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()

        # Time Features
        self.freq = freq
        # self.num_covariates = 1
        
        if self.freq=='t':          ## Currently onyl implement for seconds 't'
            self.cov = np.linspace(REGMKT_START_TIME_NS+60*10**9,REGMKT_END_TIME_NS,int(REG_N/self.sec))
            self.cov = stats.zscore(self.cov)
            self.num_covariates=2

        self.lob = lob

        print("Dataset")
        self.len = len(self.subject_map)

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수 
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        stock = row['stock']
        mean = self.avgs[self.avgs['security']==stock].reset_index()['mean_vol'+str(self.sec)][0]
        std = self.avgs[self.avgs['security']==stock].reset_index()['std_vol'+str(self.sec)][0]
        
        ### Security idx for future use (embedding security information)
        idx_security = self.securities[self.securities['Symbol']==row['stock']].index[0]

        ### Volume Sequence idx
        idx_num = row['idx_num']
        start_ind = idx_num*self.whole_len
        end_ind = (idx_num+1)*self.whole_len

        ### Load and Normalize Volume Data
        v_data = np.load(row['vol_dir'][1:])['sequence'].astype(float)
        if self.sec!=1:
            v_data = np.sum(v_data.reshape(int(v_data.shape[0]/self.sec),-1),axis=1)
        v_data = normalize_historic(v_data,mean,std)
        v_data = v_data[start_ind:end_ind]

        ### Load and Normalize LOB data
        if self.lob:
            lob_v = np.load(row['lob_dir'][1:])['sequence'].astype(float)
            lob_p = np.load(row['price_dir'][1:])['sequence'].astype(float)
            lob_v = stats.zscore(lob_v)
            lob_p = stats.zscore(lob_p)

            lob_v = lob_v[self.sec-1::self.sec]
            lob_p = lob_p[self.sec-1::self.sec]

            lob_v = lob_v[start_ind:end_ind]
            lob_p = lob_p[start_ind:end_ind]

            self.num_covariates=22 if self.peak else 21

        ### Preallocation of final data
        data = np.zeros((v_data.shape[0],self.num_covariates))
        data[:,0]=v_data

        if self.peak:
            
            # peaks = peak_identification(v_data,4,6,0.8)
            peaks1 = np.where(v_data>2,0.5,0)
            peaks2 = np.where(v_data>3,0.5,0)
            peaks = peaks1+peaks2
            # peaks = np.expand_dims(peaks["signals"], axis=1)
            data[:,1]=peaks

            if self.lob:
                # data[:,2:12] = lob_v
                # data[:,12:22] = lob_p
                data[:,2::2] = lob_p
                data[:,3::2] = lob_v

        else:
            if self.lob:
                # data[:,2:12] = lob_v
                # data[:,12:22] = lob_p
                data[:,1::2] = lob_p
                data[:,2::2] = lob_v

        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len



        data_x = data[s_begin:s_end,:]
        data_y = data[r_begin:r_end,:]
        
        
        cov=self.cov[start_ind:end_ind]
        
        data_x_mark = np.expand_dims(cov[s_begin:s_end], axis=-1)
        data_y_mark = np.expand_dims(cov[r_begin:r_end], axis=-1)

        # print(data.shape)
        
        # Xmb = normalize_max(row,self.p)
        return (data_x, data_y, data_x_mark, data_y_mark)



class Dataset_NASDAQv_Transformers(Dataset): 
    '''
    Used for Autoformer, Informer, Reformer, ...
    '''

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map, mode, start_ind, label_len, seq_len, pred_len, sec=1,freq='t',lob=True,peak=False):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        self.subject_map['idx_num'] = np.zeros((self.subject_map.shape[0]))
        # self.subject_map

        ## used for indexing the securities - unique number for securities
        self.securities = pd.read_csv('./data/NASDAQ_used/final_stocks_nasdaq.csv')

        ## Precalculated volume averages used for normalization
        self.avgs = pd.read_csv('./data/NASDAQ_used/volume_avg.csv')
        self.avgs.rename(columns = {'mean_vol':'mean_vol1', 'std_vol':'std_vol1'}, inplace = True)
        
        # use_stocks_tmp = pd.read_csv('./data/NASDAQ/nasdaq_screener_1652753301137.csv')
        # self.use_stocks = list(use_stocks_tmp['Symbol'])

        use_stocks_tmp = pd.read_csv('./data/NASDAQ_used/high_volume_stocks.csv')        # High volume stocks
        self.use_stocks = list(use_stocks_tmp['Symbols'])

        # self.use_stocks = ['AMZN','MSFT','GOOG','GOOGL','FB','NVDA','TSLA','INTC','AVGO','CSCO','COST','ASML','PEP','ADBE','TXN','QCOM','INTU','ADI','MU','LRCX','FISV','ATVI']

        self.subject_map = self.subject_map[self.subject_map['stock'].isin(self.use_stocks)].reset_index()
        self.mode = mode

        self.start_ind = start_ind
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.end_ind = start_ind+self.pred_len + self.seq_len
        
        self.sec = sec
        self.peak = peak
        # Randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()

        # Time Features
        self.freq = freq
        # self.num_covariates = 1
        
        if self.freq=='t':          ## Currently onyl implement for seconds 't'
            self.cov = np.linspace(REGMKT_START_TIME_NS+60*10**9,REGMKT_END_TIME_NS,int(REG_N/self.sec))
            self.cov = stats.zscore(self.cov)
            self.num_covariates=2

        self.lob = lob

        print("Dataset")
        self.len = len(self.subject_map)

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수 
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        stock = row['stock']
        mean = self.avgs[self.avgs['security']==stock].reset_index()['mean_vol'+str(self.sec)][0]
        std = self.avgs[self.avgs['security']==stock].reset_index()['std_vol'+str(self.sec)][0]
        
        idx = self.securities[self.securities['Symbol']==row['stock']].index[0]

        v_data = np.load(row['vol_dir'][1:])['sequence'].astype(float)
        if self.sec!=1:
            # print(v_data.shape)
            v_data = np.sum(v_data.reshape(int(v_data.shape[0]/self.sec),-1),axis=1)
            # print(v_data.shape)
        
        # v_data = normalize_max(v_data,self.p)
        # Normalize Volume with respect to historical average and mean : DONE
        v_data = normalize_historic(v_data,mean,std)

        if self.lob:
            lob_v = np.load(row['lob_dir'][1:])['sequence'].astype(float)
            lob_p = np.load(row['price_dir'][1:])['sequence'].astype(float)
            lob_v = stats.zscore(lob_v)
            lob_p = stats.zscore(lob_p)

            lob_v = lob_v[self.sec-1::self.sec]
            lob_p = lob_p[self.sec-1::self.sec]


            self.num_covariates=22 if self.peak else 21


        data = np.zeros((v_data.shape[0],self.num_covariates))
        # print(data.shape)

        data[:,0]=v_data

        if self.peak:
            
            peaks = peak_identification(v_data,4,6,0.8)
            peaks = peaks['signals']
            # peaks = np.expand_dims(peaks["signals"], axis=1)
            data[:,1]=peaks

            if self.lob:
                # data[:,2:12] = lob_v
                # data[:,12:22] = lob_p
                data[:,2::2] = lob_p
                data[:,3::2] = lob_v

        else:
            if self.lob:
                # data[:,2:12] = lob_v
                # data[:,12:22] = lob_p
                data[:,1::2] = lob_p
                data[:,2::2] = lob_v



        data = data[self.start_ind:self.end_ind,:]

        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len



        data_x = data[s_begin:s_end,:]
        data_y = data[r_begin:r_end,:]
        
        
        cov=self.cov[self.start_ind:self.end_ind]
        
        data_x_mark = np.expand_dims(cov[s_begin:s_end], axis=-1)
        data_y_mark = np.expand_dims(cov[r_begin:r_end], axis=-1)

        # print(data.shape)
        
        # Xmb = normalize_max(row,self.p)
        return (data_x, data_y, data_x_mark, data_y_mark)




class Dataset_NASDAQv_AST(Dataset): 
    '''
    Used for ASTransformer (GAN)
    '''


    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map, mode, start_ind, end_ind, p, q, sec=1,lob=False,covariates=True):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        ## used for indexing the securities - unique number for securities
        self.securities = pd.read_csv('./data/NASDAQ_used/final_stocks_nasdaq.csv')
        self.avgs = pd.read_csv('./data/NASDAQ_used/volume_avg.csv')
        self.avgs.rename(columns = {'mean_vol':'mean_vol1', 'std_vol':'std_vol1'}, inplace = True)

        
        # use_stocks_tmp = pd.read_csv('./data/NASDAQ/nasdaq_screener_1652753301137.csv')
        # self.use_stocks = list(use_stocks_tmp['Symbol'])

        use_stocks_tmp = pd.read_csv('./data/NASDAQ_used/high_volume_stocks.csv')
        self.use_stocks = list(use_stocks_tmp['Symbols'])
        
        # self.use_stocks = ['AMZN','MSFT','GOOG','GOOGL','FB','NVDA','TSLA','INTC','AVGO','CSCO','COST','ASML','PEP','ADBE','TXN','QCOM','INTU','ADI','MU','LRCX','FISV','ATVI']

        self.subject_map = self.subject_map[self.subject_map['stock'].isin(self.use_stocks)].reset_index()
        self.mode = mode
        self.p = p              ### p past
        self.q = q              ### q timeframes future
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.sec = sec

        # Randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()

        # Covariates
        self.num_covariates = 1
        if covariates:
            self.cov = np.linspace(REGMKT_START_TIME_NS+60*10**9,REGMKT_END_TIME_NS,int(REG_N/self.sec))
            self.cov = stats.zscore(self.cov)
            self.num_covariates=2

        self.lob = lob

        print("Dataset")
        self.len = len(self.subject_map)

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수 
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        stock = row['stock']

        mean = self.avgs[self.avgs['security']==stock].reset_index()['mean_vol'+str(self.sec)][0]
        std = self.avgs[self.avgs['security']==stock].reset_index()['std_vol'+str(self.sec)][0]
        
        idx = self.securities[self.securities['Symbol']==row['stock']].index[0]

        v_data = np.load(row['vol_dir'][1:])['sequence'].astype(float)
        if self.sec!=1:
            # print(v_data.shape)
            v_data = np.sum(v_data.reshape(int(v_data.shape[0]/self.sec),-1),axis=1)
            # print(v_data.shape)
        
        # v_data = normalize_max(v_data,self.p)
        # Normalize Volume with respect to historical average and mean : DONE
        v_data = normalize_historic(v_data,mean,std)

        if self.lob:
            lob_v = np.load(row['lob_dir'][1:])['sequence'].astype(float)
            lob_p = np.load(row['price_dir'][1:])['sequence'].astype(float)
            lob_v = stats.zscore(lob_v)
            lob_p = stats.zscore(lob_p)

            lob_v = lob_v[self.sec-1::self.sec]
            lob_p = lob_p[self.sec-1::self.sec]


            self.num_covariates=22


        data = np.zeros((v_data.shape[0],self.num_covariates))
        data[:,0]=v_data
        data[:,1]=self.cov
        
        if self.lob:
            # data[:,2:12] = lob_v
            # data[:,12:22] = lob_p
            data[:,2::2] = lob_p
            data[:,3::2] = lob_v

        data = data[self.start_ind:self.end_ind,:]
        # print(data.shape)
        
        # Xmb = normalize_max(row,self.p)
        return (data, idx, np.expand_dims(data[:,0], axis=-1))









class Dataset_TimeGAN(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map, mode, start_ind, end_ind, p, q):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe (지금은 그냥 data 그 자체)
        # self.subject_map

        self.mode = mode
        self.p = p
        self.q = q

        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()

        ## USE ALL DATA FOR GAN
        self.data = np.array(self.subject_map)[:,start_ind:end_ind]
        
        self.Tmb = np.arange(self.data.shape[1])[start_ind:end_ind]
        print("Dataset")
        self.len = len(self.subject_map)

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수 
    def __getitem__(self, index):
        # row1 = self.subject_map.iloc[index]
        row = self.data[index,:]
        
        # Transform data - normalize

        Xmb = normalize_max(row,self.p)
        T = self.p
        max_len = len(self.Tmb)
        return (np.expand_dims(Xmb, axis=-1), T, max_len)

