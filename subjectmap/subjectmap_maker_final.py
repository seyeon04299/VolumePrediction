
from src.constants import REGMKT_END_TIME_NS, REGMKT_START_TIME_NS, INTERVAL_SEC
from src.constants import T_N, R_N, A_N, TICKER_SYM, DATE_FROM_PATH, START_TIME_NS, END_TIME_NS, INTERVAL_NS

from collections import namedtuple
from logging import raiseExceptions
import os
import re
import pickle
import pandas as pd
import numpy as np
import numpy.linalg as la
import seaborn as sns
from typing import List

from pathlib import Path
from src.lib import Exchange
import zstandard
from pathlib import Path
import json
import numpy as np

import time

# from utils import normalize_max



"""
All half days
"""
half_days = [
    "112621"
]

"""
List of all FOMC meetings dates
"""
fomc_days = []

def normalize_max(x,p):
    '''
    Normalize with respect to max of the first p mins (ex. first 5 mins)
    '''
    max = np.amax(x[:p])
    x_normed = x/max
    return x_normed


class HistoricalData(object):
    def __init__(self, cfg, cwd="./"):
        print("Make sure to preprocess everything before running this code...")
        self.cfg = cfg  # hydra configurations
        self.csv_dir = "./csv"
        self.cwd = cwd  # current working directory (cwd)
        self.securities = self.get_securities()

        self.T = T_N                        # number of intervals
        self.D = len(self.data_dirs)   # number of days
        self.N = len(self.securities)       # number of securities

        print(f"time interval(T): {self.T}")
        print(f"number of days(D): {self.D}")
        print(f"number of securities(N): {self.N}")
        
        self.stats_map = {}
        self.subject_map = []
        
    

    def get_data_dirs(months, years):
        raise NotImplementedError()


    def get_securities(self):
        if self.cfg.dataset.security_type == "S&P500":
            df = pd.read_csv(f"{self.csv_dir}/S&P500_2022.csv")
            stock_syms = sorted(list(df["Symbol"].to_numpy()))
        else:
            raise NotImplementedError()

        stock_syms.remove('CTRA')
        stock_syms.remove('BF.B')
        stock_syms.remove('BRK.B')
        return stock_syms

    def load_actions(self, path):
        if self.exchange == Exchange.NYSE:
            return mmm.nyse.load_actions(path)
        elif self.exchange == Exchange.NASDAQ:
            return mmm.nasdaq.load_actions(path)
        else:
            raise NotImplementedError()



    def load_json_zst(self, path: Path):
        return json.loads(zstandard.decompress(Path(path).expanduser().read_bytes()))
    

    def save_volume_wLOB(self):
        # subject_map = []
        
        
        
        
        for d, data_dir in enumerate(self.data_dirs):
            print(f"computing {data_dir} volume and LOB...")
            start = time.time()

            # computing /home/jovyan/shared/axe-research/itch_data/S100421-v50/ volume and LOB...
            date = data_dir.split('/')[-2]
            nasdaq_path = data_dir.split('/')[-3]
            #print(data_dir.split('/')) #['', 'home', 'jovyan', 'shared', 'axe-research', 'itch_data', 'S090121-v50', '']
            
            
            ## Create npz save folder if not exist
            folder_dir = os.path.join('/home/jovyan/shared/axe-research','npz_2109_2203',date)
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)
            
            
            ## For each Securities
            for n, sym in enumerate(self.securities):

                if n%50==0:
                    print("Computed {}/{} stocks...".format(n,len(self.securities)))
                sym_stats_map_tmp = {}
                
                #### GET VOLUME, LOB ####
                try:
                    market_stat = self.load_json_zst(data_dir + f"{sym}.json.zst")
                except Exception as e:
                    # TODO XXX
                    # this means that sym is not in market stats
                    # most likely because the stock is listed in NYSE and
                    # not traded in itch
                    print(f"{sym} data not available in {data_dir}. This is most likely due to the fact that some NYSE stocks do not get traded in Nasdaq")
                    print("This should not print")
                    print(e)
                    raiseExceptions()
                
                ### Load Volume and LOB
                vol = np.array(market_stat["interval_volume"])
                lob = np.array(market_stat["lob_level_5"],dtype=float)
                
                ### Regular market times
                vol = vol[R_N:A_N]
                lob = lob[R_N:A_N,1:]     ## Remove Timestamp

                ### For finding mean&std of volume with n seconds interval (instead of 1sec)
                vol2 = np.sum(vol.reshape(int(vol.shape[0]/2),-1),axis=1)
                vol5 = np.sum(vol.reshape(int(vol.shape[0]/5),-1),axis=1)
                vol10 = np.sum(vol.reshape(int(vol.shape[0]/10),-1),axis=1)
                vol20 = np.sum(vol.reshape(int(vol.shape[0]/20),-1),axis=1)
                vol30 = np.sum(vol.reshape(int(vol.shape[0]/30),-1),axis=1)
                vol60 = np.sum(vol.reshape(int(vol.shape[0]/60),-1),axis=1)
                
                ### Swap volume and price for bids
                lob[:, [10,11,12,13,14,15,16,17,18,19]] = lob[:, [11,10,13,12,15,14,17,16,19,18]]   
                lob[:,1::2]=np.true_divide(lob[:,1::2], 10000)
                
                ### Sort LOB
                shapes = lob.shape
                lob = lob.reshape(shapes[0],-1,2)
                for i in range(shapes[0]):
                    lob[i] = lob[i][lob[i][:,1].argsort()]
                lob = lob.reshape(shapes)
                
                ### Calculate S1 = sum(x_i), S2 = sum(x_i^2) for LOB and volume (used later for calculating historical mean and std of volume and lob for each stocks)
                lob_v_S1 = np.sum(lob[:,0::2])
                lob_p_S1 = np.sum(lob[:,1::2])
                lob_v_S2 = np.sum(lob[:,0::2]**2)
                lob_p_S2 = np.sum(lob[:,1::2]**2)
                sym_stats_map_tmp['lob_v_S1'],sym_stats_map_tmp['lob_p_S1'], sym_stats_map_tmp['lob_v_S2'], sym_stats_map_tmp['lob_p_S2'] = lob_v_S1, lob_p_S1, lob_v_S2, lob_p_S2
                sym_stats_map_tmp['n_lob'] = lob.shape[0]*lob.shape[1]/2
                
                vol_S1 = np.sum(vol)
                vol_S2 = np.sum(vol**2)
                sym_stats_map_tmp['vol_S1'],sym_stats_map_tmp['vol_S2'] = vol_S1, vol_S2
                sym_stats_map_tmp['n_vol'] = vol.shape[0]
                
                vols = [vol2,vol5,vol10,vol20,vol30,vol60]
                for indx, sec in enumerate(['2','5','10','20','30','60']):
                    name_S1 = 'vol'+sec+'_S1'
                    name_S2 = 'vol'+sec+'_S2'
                    name_nvol = 'n_vol'+sec
                    vol_ = vols[indx]
                    vol_S1 = np.sum(vol_)
                    vol_S2 = np.sum(vol_**2)
                    sym_stats_map_tmp[name_S1],sym_stats_map_tmp[name_S2] = vol_S1, vol_S2
                    sym_stats_map_tmp[name_nvol] = vol_.shape[0]
                
                if d == 0:
                    self.stats_map[str(sym)] = sym_stats_map_tmp
                else:
                    for item in sym_stats_map_tmp.keys():
                        self.stats_map[str(sym)][item]+=sym_stats_map_tmp[item]
                
                ### SAVE npz files
                vol_dir = os.path.join(folder_dir,'vol_'+sym)
                lob_dir = os.path.join(folder_dir,'lob_'+sym)
                np.savez(vol_dir, sequence=vol)
                np.savez(lob_dir, sequence=lob)
                self.subject_map.append({
                    "date": date,
                    "stock":sym,
                    'vol_dir': vol_dir+'.npz',
                    'lob_dir': lob_dir+'.npz'
                })
            end = time.time()
            print ("Time elapsed for calculating volume/lob (with mean/std...) of 1 day:", end - start)

        subject_map = pd.DataFrame(self.subject_map)
        subject_map.to_csv('./subjectmap_2109_2203.csv',index=False)
        print('Subject map saved to "./subjectmap_2109_2203.csv" ')


    def sym_historical_stats_calculator(self):
        print('Begin Historical Mean and Std Calculator')
        historical_stats = []
        for sym in self.stats_map.keys():
            sd = self.stats_map[sym]
            
            lob_v_mean = sd['lob_v_S1']/sd['n_lob']
            lob_v_std = np.sqrt(sd['lob_v_S2']/sd['n_lob']-(sd['lob_v_S1']/sd['n_lob'])**2)

            lob_p_mean = sd['lob_p_S1']/sd['n_lob']
            lob_p_std = np.sqrt(sd['lob_p_S2']/sd['n_lob']-(sd['lob_p_S1']/sd['n_lob'])**2)
            
            vol_mean = sd['vol_S1']/sd['n_vol']
            vol_std = np.sqrt(sd['vol_S2']/sd['n_vol']-(sd['vol_S1']/sd['n_vol'])**2)
            
            vol2_mean = sd['vol2_S1']/sd['n_vol2']
            vol2_std = np.sqrt(sd['vol2_S2']/sd['n_vol2']-(sd['vol2_S1']/sd['n_vol2'])**2)

            vol5_mean = sd['vol5_S1']/sd['n_vol5']
            vol5_std = np.sqrt(sd['vol5_S2']/sd['n_vol5']-(sd['vol5_S1']/sd['n_vol5'])**2)
            
            vol10_mean = sd['vol10_S1']/sd['n_vol10']
            vol10_std = np.sqrt(sd['vol10_S2']/sd['n_vol10']-(sd['vol10_S1']/sd['n_vol10'])**2)
            
            vol20_mean = sd['vol20_S1']/sd['n_vol20']
            vol20_std = np.sqrt(sd['vol20_S2']/sd['n_vol20']-(sd['vol20_S1']/sd['n_vol20'])**2)
            
            vol30_mean = sd['vol30_S1']/sd['n_vol30']
            vol30_std = np.sqrt(sd['vol30_S2']/sd['n_vol30']-(sd['vol30_S1']/sd['n_vol30'])**2)
            
            vol60_mean = sd['vol60_S1']/sd['n_vol60']
            vol60_std = np.sqrt(sd['vol60_S2']/sd['n_vol60']-(sd['vol60_S1']/sd['n_vol60'])**2)
            
            historical_stats.append( {
                'security' : sym,
                'lob_v_mean' : lob_v_mean,
                'lob_v_std' : lob_v_std,
                'lob_p_mean' : lob_p_mean,
                'lob_p_std' : lob_p_std,
                'vol_mean' : vol_mean,
                'vol_std' : vol_std,
                'vol2_mean' : vol2_mean,
                'vol2_std' : vol2_std,
                'vol5_mean' : vol5_mean,
                'vol5_std' : vol5_std,
                'vol10_mean' : vol10_mean,
                'vol10_std' : vol10_std,
                'vol20_mean' : vol20_mean,
                'vol20_std' : vol20_std,
                'vol30_mean' : vol30_mean,
                'vol30_std' : vol30_std,
                'vol60_mean' : vol60_mean,
                'vol60_std' : vol60_std,
            })
            
        
        historical_stats = pd.DataFrame(historical_stats)
        historical_stats.to_csv('../data/NASDAQ_used/historical_stats_2109_2203.csv',index=False)
        print('historical_stats saved to "../data/NASDAQ_used/historical_stats_2109_2203.csv" ')
        




class NasdaqData(HistoricalData):
    def __init__(self, cfg, cwd="../data"):
        self.exchange = Exchange.NASDAQ
        self.data_dir = '/home/jovyan/shared/axe-research/itch_data/'

        self.data_dirs = self.get_data_dirs(
                months=["01","02","03","09","10","11","12"],
                years=["21","22"]
            )
        print(f"itch data_dirs: {self.data_dirs}")
        super().__init__(cfg, cwd)


    """
    Finds all itch data directories that we want to use
    """
    def get_data_dirs(self,
                      months: List[str],
                      years: List[str]):

        itch_data_dirs = []
        for y in years:
            for m in months:
                pattern = re.compile(f"S{m}[0-9][0-9]{y}-v50$")  # e.g. 090121-v50
                # add all itch directories that match with our pattern
                itch_data_dirs += sorted([f"{self.data_dir}" + x + "/"
                            for x in os.listdir(f"{self.data_dir}") if pattern.match(x)])

        # remove all half days
        for half_day in half_days:
            data_dir = f"{self.data_dir}S" + half_day + "-v50/"
            print('halfdir : ', data_dir)
            if data_dir in itch_data_dirs:
                itch_data_dirs.remove(data_dir)

        # remove all fomc dates
        for fomc_day in fomc_days:
            data_dir = f"{self.data_dir}S" + fomc_day + "-v50/"
            if data_dir in itch_data_dirs:
                itch_data_dirs.remove(data_dir)

        return itch_data_dirs


if __name__ == '__main__':
    DATASET = namedtuple("dataset", ["start_date", "end_date", "security_type", "reset"])
    ARGS = namedtuple("cfg", ["dataset"])
    cfg = ARGS(DATASET("090121", "033122", "S&P500", True))
    


    nasdaq_data = NasdaqData(cfg)
    nasdaq_data.save_volume_wLOB()
    nasdaq_data.sym_historical_stats_calculator()