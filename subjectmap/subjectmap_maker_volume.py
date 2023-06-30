
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

import mmm.nasdaq
import mmm.nyse

from mmm.nasdaq import compile_trajectory_with_volume_level

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

    def load_json_zst(self, path):
        if self.exchange == Exchange.NYSE:
            return mmm.nyse.load_json_zst(path)
        elif self.exchange == Exchange.NASDAQ:
            return mmm.nasdaq.load_json_zst(path)
        else:
            raise NotImplementedError()

    
    def compute_volume_wLOB_data(self, lob_level):
        """
        vols: volume of all stocks. each element represents the volume of stock n
              on day d, and on time interval t.
        nd_vols: non displayable volume of all stocks. each element represents the volume
              of stock n on day d and on time interval t.
        lp_vols: market marker's volume of all stocks. each element represents the volume
              of stock n on day d and on time interval t.
        """
        vols = np.zeros((self.D, self.N, self.T))
        nd_vols = np.zeros((self.D, self.N, self.T))
        lp_vols = np.zeros((self.D, self.N, self.T))

        interval_execute_msg_count = np.zeros((self.D, self.N, self.T))
        

        subject_map = []
        for d, data_dir in enumerate(self.data_dirs):
            print(f"computing {data_dir} volume and LOB...")
            date = data_dir.split('/')[-2]
            nasdaq_path = data_dir.split('/')[-3]


            ## Create npz save folder if not exist
            folder_dir = os.path.join('/home/jovyan/shared/axe-research/','npz_files_whole',date)
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)


            ## For Each Securities
            for n, sym in enumerate(self.securities):
                
                ##############################
                ##### Compute intraday Volume
                ##############################

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

                if self.exchange == Exchange.NASDAQ:
                    vol = np.array(market_stat["interval_volume_printable"]).ravel()
                    
                    lp_vol = np.array(market_stat["interval_lp_volume_printable"]).ravel()
                    
                elif self.exchange == Exchange.NYSE:
                    vol = np.array(market_stat["interval_volume_all"]).ravel()
                    lp_vol = np.array(market_stat["interval_lp_volume_all"]).ravel()
                else:
                    raise NotImplementedError()
                
                
                # vol = normalize_max(vol,300*INTERVAL_SEC)
                vol = vol[R_N:A_N]
                # lp_vol = normalize_max(lp_vol,300*INTERVAL_SEC)
                lp_vol = lp_vol[R_N:A_N]

                vol_dir = os.path.join(folder_dir,'vol_'+sym)
                lp_vol_dir = os.path.join(folder_dir,'lp_vol_'+sym)

                # vols[d, n, :vol.shape[0]] = vol
                # lp_vols[d, n, :lp_vol.shape[0]] = lp_vol

                nd_vol = np.array(market_stat["interval_nondisp_volume"]).ravel()
                # nd_vol = normalize_max(nd_vol,300*INTERVAL_SEC)
                nd_vol = nd_vol[R_N:A_N]


                nd_vol_dir = os.path.join(folder_dir,'nd_vol_'+sym)
                # nd_vols[d, n, :nd_vol.shape[0]] = nd_vol

                count = np.array(market_stat["interval_execute_msg_count_all"]).ravel()
                interval_execute_msg_count[d, n, :count.shape[0]] = count

                # assert not(np.all(vols[d, n, :] == 0)), f"{sym}, {vols[d, n, :]}, {market_stat}"

                ##############################
                ##### Compute Intraday LOB
                ##############################
                
                path = f"{data_dir}{sym}.bin.zst"
                actions = self.load_actions(path)
                target_times = np.arange(REGMKT_START_TIME_NS, REGMKT_END_TIME_NS, INTERVAL_NS)

                assert target_times.shape[0] == A_N-R_N, f"target times shape: {target_times.shape},  A_N-R_N: {A_N-R_N}"
                # note that np.searchsorted can place objects at the beginning and the end of
                # our actions. If it places at the end, it will cause index out of bounds error
                # so for those indicies, we simply subtract by 1.
                target_indicies = np.searchsorted(actions[:, 1], target_times,side='left') -1
                target_indicies[target_indicies<0] = 0
                target_indicies[target_indicies == actions.shape[0]] = actions.shape[0]-1

                assert np.all(target_indicies < actions.shape[0]), \
                f"target indices out of bounds {actions.shape[0]} {target_indicies[target_indicies >= actions.shape[0]]}"

                trajectories = compile_trajectory_with_volume_level(actions,
                                                        target_indicies,
                                                        [0 for _ in range(len(target_indicies))],
                                                        level=lob_level,
                                                        is_inclusive=True,
                                                        )
                
                lob = np.zeros((len(trajectories),lob_level*2))
                prices = np.zeros((len(trajectories),lob_level*2))
                
                for i, traj in enumerate(trajectories):
                    
                    ask_dict = traj[3]['Ask']
                    bid_dict = traj[3]['Bid']
                    
                    askp = np.array(sorted(ask_dict, reverse=True))
                    askv = np.array([ask_dict[k] for k in sorted(ask_dict, reverse=True)])
                    
                    bidp = np.array(sorted(bid_dict, reverse=True))
                    bidv = np.array([bid_dict[k] for k in sorted(bid_dict, reverse=True)])

                    if len(askp)<5:
                        askv = np.pad(askv, ( 5 - len(askv)%5,0), 'constant')
                        if len(askp)==0:
                            print(i, sym, "Empty ask")
                            askp = np.pad(askp, ( 5 - len(askp)%5,0), 'constant',constant_values=(prices[i-1,4]))
                        else:
                            askp = np.pad(askp, ( 5 - len(askp)%5,0), 'constant',constant_values=(askp[0]+100))

                    if len(bidp)<5:
                        bidv = np.pad(bidv, ( 0, 5 - len(bidv)%5), 'constant')
                        if len(bidp)==0:
                            print(i, sym, "Empty bid")
                            bidp = np.pad(bidp, ( 0, 5 - len(bidp)%5), 'constant',constant_values=(prices[i-1,5]))
                        else:
                            bidp = np.pad(bidp, ( 0, 5 - len(bidp)%5), 'constant',constant_values=(bidp[-1]-100))

                    lob[i,:5]=askv
                    lob[i,5:]=bidv
                    prices[i,:5]=askp
                    prices[i,5:]=bidp

                lob_dir = os.path.join(folder_dir,'lob'+sym)
                price_dir = os.path.join(folder_dir,'price'+sym)
                np.savez(vol_dir, sequence=vol)
                np.savez(lp_vol_dir, sequence=lp_vol)
                np.savez(nd_vol_dir, sequence=nd_vol)
                np.savez(lob_dir, sequence=lob)
                np.savez(price_dir, sequence=prices)
                subject_map.append({
                    "date": date,
                    "stock":sym,
                    'vol_dir': vol_dir+'.npz',
                    'lp_vol_dir': lp_vol_dir+'.npz',
                    'nd_vol_dir': nd_vol_dir+'.npz',
                    'lob_dir': lob_dir+'.npz',
                    'price_dir': price_dir+'.npz'
                    
                })

                
                

                ##### MidPrice
                # bbo = self.load_bbo(data_dir + f"{sym}_bbo.bin.zst")
        subject_map = pd.DataFrame(subject_map)
        subject_map.to_csv('./subjectmap_volumeLOB.csv',index=False)
        # self.vols = vols
        # self.lp_vols = lp_vols
        # self.nd_vols = nd_vols

        # """
        # Note that we subtract self.T by 1. This is because we have an extra interval from 20:00 to 20:01 
        # because nyse taq data have messages after 20:00. 
        # """
        # self.interval_execute_msg_count = interval_execute_msg_count

        # self.regmkt_vols = self.vols[:, :, R_N:A_N]
        # self.regmkt_nd_vols = self.nd_vols[:, :, R_N:A_N]
        # self.regmkt_lp_vols = self.lp_vols[:, :, R_N:A_N]

        # self.regmkt_interval_execute_msg_count = self.interval_execute_msg_count[:, :, R_N:A_N]
        # self.regmkt_avg_touch_size = np.nan_to_num(self.regmkt_vols / self.regmkt_interval_execute_msg_count, nan=0.0)








class NasdaqData(HistoricalData):
    def __init__(self, cfg, cwd='/home/jovyan/shared/axe-research/'):
        self.exchange = Exchange.NASDAQ
        self.data_dir = f"{cwd}/itch_data/"  # data directory

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
    cfg = ARGS(DATASET("090121", "123121", "S&P500", True))
    


    nasdaq_data = NasdaqData(cfg)
    nasdaq_data.compute_volume_wLOB_data(5)