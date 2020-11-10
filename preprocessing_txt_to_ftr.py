import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import os

if __name__ == '__main__':
    # data 폴더 생성
    if not os.path.isdir('data'):
        os.mkdir('data')
        
    # txt 파일 위치 로드, 상황에 맞게 변경해야 함.
    txt = glob('../data/*.txt')

    # convert from txt to ftr
    for t in tqdm(txt):
        name = t.split('/')[-1].split('_')[0]
        if not os.path.isfile('./data/%s.ftr' % name):
            df = pd.read_csv(t, sep='\t').iloc[:,:-1]
            df.to_feather('./data/%s.ftr' % name)
        else:
            print('%s.ftr already exists.' % name)

    # Our dataset 201912 ~ 202007
    name_list = ['201912', '202001', '202002', '202003', '202004', '202005', '202006', '202007']

    # ftr file -> our data structure (front, log)
    for name  in tqdm(name_list):
        if not os.path.isfile('./data/%s_front.ftr' % name):
            df = pd.read_feather('./data/%s.ftr' % name)
            df.iloc[:,:4].to_feather('./data/%s_front.ftr' % name)
            df = df.iloc[:,4:10004].rolling(5, axis=1).mean().iloc[:,4:]
            log_df = np.log10(df)
            log_df.to_feather('./data/log_%s.ftr'% name)
        else:
            print('%s front and log already exists.' % name)

    if not os.path.isfile('data/120102_front.ftr'):
        front1 = pd.read_feather('data/201912_front.ftr')
        hz1 = pd.read_feather('data/log_201912.ftr')
        front2 = pd.read_feather('data/202001_front.ftr')
        hz2 = pd.read_feather('data/log_202001.ftr')
        front3 = pd.read_feather('data/202002_front.ftr')
        hz3 = pd.read_feather('data/log_202002.ftr')


        front = pd.concat((front1, front2, front3), axis=0)
        hz = pd.concat((hz1, hz2, hz3), axis=0)
        front.reset_index(drop=True, inplace=True)
        hz.reset_index(drop=True, inplace=True)

        front.to_feather(('data/120102_front.ftr'))
        hz.to_feather(('data/log_120102.ftr'))
        
    else:
        print('120102 front and log already exists.')