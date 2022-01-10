from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torch
import pandas as pd

# class 0 and 1 respectively
def init__weighted_random__sampler(Y):
    """
    this sampler helps to deal with class imbalance 
    """
    count=Counter(Y)
    class_count=np.array([count[0],count[1]]) 
    weight=1./class_count
    samples_weight = np.array([weight[t] for t in Y])
    samples_weight=torch.from_numpy(samples_weight)
    weighted_random_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return weighted_random_sampler


def get_callbacks():
    ea_stop, checkpoint = EarlyStopping(monitor='val_loss', patience=5, mode='min'), ModelCheckpoint(
        f'models/new_models_with_top_3k',
        filename='{epoch}-{val_loss:.3f}',
        monitor='val_loss',
        save_top_k=3)
    callbacks = ea_stop
    return callbacks, checkpoint


def scale_data_without_nan_and_fillna_after(df, numerical_cols, numerical_col_minmaxscale=None):
    if numerical_col_minmaxscale == None:
        numerical_col__minmaxscaler = {} 
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col])
        if numerical_col_minmaxscale == None:
            scaler = MinMaxScaler()
        nan_indexes = df[col].index[df[col].apply(np.isnan)]
        no_nan_indexes = df[df[col].notnull()].index
        if numerical_col_minmaxscale == None:
            df.loc[no_nan_indexes, col] = scaler.fit_transform(df.iloc[no_nan_indexes][col].values.reshape(-1, 1))
            numerical_col__minmaxscaler[col] = scaler
        else:
            df.loc[no_nan_indexes, col] = numerical_col_minmaxscale[col].transform(df.iloc[no_nan_indexes][col].values.reshape(-1, 1))
        df.loc[nan_indexes, col] = -1
    if numerical_col_minmaxscale == None:
        return numerical_col__minmaxscaler
    
    
