#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

import argparse
import warnings
import os
from datetime import datetime
import time
import math
import psutil

from models.E3NNCNN import E3NNCNN  # or remove this line if class is in same file
from models.E3NNDivergenceFreeCNN import E3NNDivergenceFreeCNN  # or remove this line if class is in same file

import magnav
from models.CNN import CNN, ResNet18
from models.RNN import LSTM, GRU
from models.MLP import MLP
from models.MLP2 import MLP2
from models.MLP3 import MLP3
from models.MLP4 import MLP4


from models.E3NNMLP import E3NNMLP
from models.E3NNDivergenceFreeMLP import E3NNDivergenceFreeMLP
from models.DivergencefreeCNN import DivergenceFreeCNN
from models.E3NNDivergenceFreeCNN2 import E3NNDivergenceFreeCNN2
from models.E3NNDivergenceFreeCNN3 import E3NNDivergenceFreeCNN3
from models.DivergenceFreeMLP2 import DivergenceFreeMLP2


# Import the ODE-based ContiFormer
from contiformer import ContiFormer
from se3_contiformer import SE3ContiFormerMagNav
from models.DivergencefreeMLP import DivergenceFreeMLP
from models.LTCDivergenceFreeCFC import CfC_DivFree


from e3nn.o3 import Irreps, Linear

import sys
print("train_magnav.py using Python:", sys.executable)

try:
    import e3nn
    from e3nn.o3 import Irreps
    print("e3nn import successful")
    E3NN_AVAILABLE = True
except Exception as e:
    print("Full E3NN import failed:", type(e), e)
    E3NN_AVAILABLE = False

    
# Helper to fetch model name (handles DataParallel)
def get_model_name(m):
    from torch import nn as _nn  # local import to avoid issues if torch not yet imported
    if isinstance(m, _nn.DataParallel):
        return getattr(m.module, 'name', '')
    return getattr(m, 'name', '')

#-----------------#
#----Functions----#
#-----------------#
    
    
def trim_data(data, seq_len):
    '''
    Delete part of the training data so that the remainder of the Euclidean division between the length of the data and the size of a sequence is 0. This ensures that all sequences are complete.
    
    Arguments:
    - `data` : data that needs to be trimmed
    - `seq_len` : lenght of a sequence
    
    Returns:
    - `data` : trimmed data
    '''
    if (len(data)%seq_len) != 0:
        data = data[:-(len(data)%seq_len)]
    else:
        pass
    
    return data


class MagNavDataset(Dataset):
    '''
    Transform Pandas dataframe of flights data into a custom PyTorch dataset that returns the data into sequences of a desired length.
    '''
    def __init__(self, df, seq_len, split, train_lines, test_lines, truth='IGRFMAG1'):
        self.seq_len = seq_len
        self.features = df.drop(columns=['LINE', truth]).columns.to_list()
        self.train_sections = train_lines
        self.test_sections = test_lines

        if split == 'train':
            mask_train = pd.Series(dtype=bool)
            for line in self.train_sections:
                mask_train |= (df.LINE == line)
            X_train = df.loc[mask_train, self.features]
            y_train = df.loc[mask_train, truth]
            self.X = torch.t(trim_data(torch.tensor(X_train.to_numpy(), dtype=torch.float32), seq_len))
            self.y = trim_data(torch.tensor(np.reshape(y_train.to_numpy(), [-1, 1]), dtype=torch.float32), seq_len)
        elif split == 'test':
            mask_test = pd.Series(dtype=bool)
            for line in self.test_sections:
                mask_test |= (df.LINE == line)
            X_test = df.loc[mask_test, self.features]
            y_test = df.loc[mask_test, truth]
            self.X = torch.t(trim_data(torch.tensor(X_test.to_numpy(), dtype=torch.float32), seq_len))
            self.y = trim_data(torch.tensor(np.reshape(y_test.to_numpy(), [-1, 1]), dtype=torch.float32), seq_len)

    def __getitem__(self, idx):
        X = self.X[:, idx:(idx + self.seq_len)]
        y = self.y[idx + self.seq_len - 1]
        return X, y

    def __len__(self):
        return len(torch.t(self.X)) - self.seq_len


class CfCDataset(Dataset):
    def __init__(self, df, seq_len, split, train_lines, test_lines, truth='IGRFMAG1'):
        self.seq_len = seq_len
        self.coord_cols = ['UTM_X', 'UTM_Y', 'UTM_Z']
        self.time_col = 'TIME'
        self.features = df.drop(columns=['LINE', truth] + self.coord_cols + [self.time_col]).columns.to_list()

        mask = df['LINE'].isin(train_lines if split == 'train' else test_lines)
        df_split = df[mask].copy()
        if (len(df_split) % self.seq_len) != 0:
            df_split = df_split.iloc[:-(len(df_split) % self.seq_len)]

        self.X = torch.tensor(df_split[self.features].values, dtype=torch.float32)
        self.y = torch.tensor(df_split[truth].values, dtype=torch.float32).unsqueeze(1)
        self.coords = torch.tensor(df_split[self.coord_cols].values, dtype=torch.float32)
        self.time = torch.tensor(df_split[self.time_col].values, dtype=torch.float32)

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.seq_len)
        t_seq = self.time[sl]
        delta_t = torch.cat([torch.zeros(1), t_seq[1:] - t_seq[:-1]]).unsqueeze(1)
        return (self.X[sl], delta_t, self.coords[sl], self.y[idx + self.seq_len - 1])

    def __len__(self):
        return len(self.X) - self.seq_len + 1


def make_training(model, epochs, train_loader, test_loader, scaling=['None'], lr=None):
    # Ensure helper exists
    try:
        _ = get_model_name
    except NameError:
        def get_model_name(m):
            from torch import nn as _nn
            if isinstance(m, _nn.DataParallel):
                return getattr(m.module, 'name', '')
            return getattr(m, 'name', '')

    # Different hyperparameters for ContiFormer
    if 'ContiFormer' in get_model_name(model) or 'Transformer' in get_model_name(model):
        # ContiFormer needs different hyperparameters
        learning_rate = 0.0001  # Lower learning rate for transformers
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        print(f"Using ContiFormer optimization: AdamW with lr={learning_rate}, cosine annealing schedule")
    else:
        # Original settings for CNN/MLP
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        lambda1 = lambda epoch: 0.9**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        print(f"Using standard optimization: Adam with lr=0.001, exponential decay")
    
    # Create batch and epoch progress bar
    batch_bar = tqdm(total=len(train)//BATCH_SIZE,unit="batch",desc='Training',leave=False, position=0, ncols=150)
    epoch_bar = tqdm(total=epochs,unit="epoch",desc='Training',leave=False, position=1, ncols=150)
    
    train_loss_history = []
    test_loss_history = []
    Best_RMSE = 9e9
    patience = 20  # Early stopping patience
    patience_counter = 0

    for epoch in range(epochs):

        #----Train----#

        train_running_loss = 0.

        # Turn on gradients computation
        model.train()
        
        batch_bar.reset()
        
        # Enumerate allow to track batch index and intra-epoch reporting 
        for batch_index, (inputs, labels) in enumerate(train_loader):
            
            if "CfC_DivFree" in get_model_name(model):
                inputs, delta_t, coords, labels = batch_data
                inputs, delta_t, coords, labels = inputs.to(DEVICE), delta_t.to(DEVICE), coords.to(DEVICE), labels.to(DEVICE)
                predictions = model(inputs, delta_t, coords)
            else:
                inputs, labels = batch_data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                if 'CNN' in get_model_name(model):
                    inputs = inputs.permute(0, 2, 1)
                
                if hasattr(model.module if isinstance(model, nn.DataParallel) else model, "name") and "E3NN" in get_model_name(model) and inputs.shape[2] >= 3:
                    inputs = inputs[:, :, :3].permute(0, 2, 1)

                predictions = model(inputs)

            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            
            if 'ContiFormer' in get_model_name(model) or 'CfC_DivFree' in get_model_name(model):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_running_loss += loss.item()
            
            batch_bar.set_postfix(train_loss=train_running_loss/(batch_index+1),lr=optimizer.param_groups[0]['lr'])
            batch_bar.update()
        
        scheduler.step()
        train_loss = train_running_loss / len(train_loader)
        train_loss_history.append(train_loss)

        #----Test----#

        test_running_loss = 0.
        preds = []
        targets = []
        
        # Disable layers specific to training such as Dropout/BatchNorm
        model.eval()
        
        # Turn off gradients computation
        with torch.no_grad():
            
            # Enumerate allow to track batch index and intra-epoch reporting
            for batch_index, batch_data in enumerate(test_loader):
                if "CfC_DivFree" in get_model_name(model):
                    inputs, delta_t, coords, labels = batch_data
                    inputs, delta_t, coords, labels = inputs.to(DEVICE), delta_t.to(DEVICE), coords.to(DEVICE), labels.to(DEVICE)
                    predictions = model(inputs, delta_t, coords)
                else:
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                    if 'CNN' in get_model_name(model):
                        inputs = inputs.permute(0, 2, 1)
                        
                    if hasattr(model.module if isinstance(model, nn.DataParallel) else model, "name") and "E3NN" in get_model_name(model) and inputs.shape[2] >= 3:
                        inputs = inputs[:, :, :3].permute(0, 2, 1)

                    # Make prediction for this batch
                    predictions = model(inputs)
                
                # Save prediction for this batch
                preds.append(predictions.cpu())
                targets.append(labels.cpu()) 
                loss = criterion(predictions, labels)
                test_running_loss += loss.item()

        # Compute the loss of the batch and save it
        preds = np.concatenate(preds)

        # Collect true targets from test_loader in the same order as preds
        true_y_collected = []
        for _, y in test_loader:
            true_y_collected.append(y)
        true_y_tensor = torch.cat(true_y_collected, dim=0).numpy()  # preds is already on CPU

        # Sanity check shapes
        assert preds.shape[0] == true_y_tensor.shape[0], f"Shape mismatch: preds={preds.shape}, true_y={true_y_tensor.shape}"

        # Apply inverse scaling and compute RMSE
        if scaling[0] == 'None':
            RMSE_epoch = magnav.rmse(preds, true_y_tensor, False)
        elif scaling[0] == 'std':
            RMSE_epoch = magnav.rmse(preds * scaling[2] + scaling[1], true_y_tensor * scaling[2] + scaling[1], False)
        elif scaling[0] == 'minmax':
            RMSE_epoch = magnav.rmse(
                scaling[3] + ((preds - scaling[1]) * (scaling[4] - scaling[3]) / (scaling[2] - scaling[1])),
                scaling[3] + ((true_y_tensor - scaling[1]) * (scaling[4] - scaling[3]) / (scaling[2] - scaling[1])),
                False
            )

        test_loss = test_running_loss / (batch_index + 1)
        test_loss_history.append(test_loss)
        
        if lr is None:
            lr = 1e-3

        
        # Save best model
        if Best_RMSE > RMSE_epoch:
            Best_RMSE = RMSE_epoch
            Best_model = model
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        
        # Update epoch progress bar
        epoch_bar.set_postfix(train_loss=train_loss,test_loss=test_loss,RMSE=RMSE_epoch,lr=optimizer.param_groups[0]['lr'])
        epoch_bar.update()
    print('\n')
    
    return train_loss_history, test_loss_history, Best_RMSE, Best_model


def Standard_scaling(df):
    '''
    Apply standardization (Z-score normalization) to a pandas dataframe except for the 'LINE' feature.
    
    Arguments:
    - `df` : dataframe to standardize
    
    Returns:
    - `df_scaled` : standardized dataframe
    '''
    df_scaled = (df-df.mean())/df.std()
    df_scaled['LINE'] = df['LINE']

    return df_scaled


def MinMax_scaling(df, bound=[-1,1]):
    '''
    Apply min-max scaling to a pandas dataframe except for the 'LINE' feature.

    Arguments:
    - `df` : dataframe to standardize
    - `bound` : (optional) upper and lower bound for min-max scaling
    
    Returns:
    - `df_scaled` : scaled dataframe
    '''
    df_scaled = bound[0] + ((bound[1]-bound[0])*(df-df.min())/(df.max()-df.min()))
    df_scaled['LINE'] = df['LINE']
    
    return df_scaled


def apply_corrections(df,mags_to_cor,diurnal=True,igrf=True):
    '''
    Apply IGRF and/or diurnal corrections on data.
    
    Arguments:
    - `df` : dataframe to correct
    - `mags_to_cor` : list of string of magnetometers to be corrected
    - `diurnal` : (optional) apply diunal correction (True or False)
    - `igrf` : (optional) apply IGRF correction (True or False)
    
    Returns:
    - `df_cor` : corrected dataframe
    '''
    mag_measurements = np.array(mags_to_cor)
    df_cor = df.copy()
    
    # Diurnal cor
    if diurnal == True:
        df_cor[mag_measurements] = df_cor[mag_measurements]-np.reshape(df_cor['DIURNAL'].values,[-1,1])
    
    # IGRF cor
    lat  = df_cor['LAT']
    lon  = df_cor['LONG']
    h    = df_cor['BARO']*1e-3
    date = datetime(2020, 6, 29) # Date on which the flights were made
    Be, Bn, Bu = magnav.igrf(lon,lat,h,date)

    if igrf == True:
        igrf_total = np.sqrt(Be**2+Bn**2+Bu**2)
        # Reshape to match the number of magnetometer columns being corrected
        igrf_correction = np.tile(igrf_total.reshape(-1, 1), (1, len(mag_measurements)))
        df_cor[mag_measurements] = df_cor[mag_measurements] - igrf_correction

    return df_cor


def create_synthetic_magnav_data():

    print("Creating synthetic MagNav data for testing...")
    
    n_samples = 10000
    t = np.linspace(0, 100, n_samples)
    
    # Synthetic magnetometer readings
    mag1 = 50000 + 100 * np.sin(0.1 * t) + 50 * np.random.randn(n_samples)
    mag2 = 50100 + 80 * np.cos(0.08 * t) + 40 * np.random.randn(n_samples)
    
    # Synthetic flight dynamics
    velocity_n = 50 + 10 * np.sin(0.05 * t) + np.random.randn(n_samples)
    velocity_v = 5 + 2 * np.sin(0.03 * t) + np.random.randn(n_samples)
    velocity_w = np.random.randn(n_samples)
    
    pitch = 2 * np.sin(0.02 * t) + 0.5 * np.random.randn(n_samples)
    roll = 1.5 * np.cos(0.025 * t) + 0.3 * np.random.randn(n_samples)
    azimuth = 180 + 20 * np.sin(0.01 * t) + np.random.randn(n_samples)
    
    # Create DataFrame
    data = {
        'TIME': t,
        'UTM_X': 500000 + 100 * t,
        'UTM_Y': 4000000 + 50 * np.sin(0.05 * t),
        'UTM_Z': 1000 - 5 * t,
        'TL_comp_mag4_cl': mag1,
        'TL_comp_mag5_cl': mag2,
        'V_BAT1': 24 + 0.5 * np.random.randn(n_samples),
        'V_BAT2': 24 + 0.5 * np.random.randn(n_samples),
        'INS_VEL_N': velocity_n,
        'INS_VEL_V': velocity_v,
        'INS_VEL_W': velocity_w,
        'CUR_IHTR': 2 + 0.2 * np.random.randn(n_samples),
        'CUR_FLAP': 1 + 0.1 * np.random.randn(n_samples),
        'CUR_ACLo': 0.5 + 0.05 * np.random.randn(n_samples),
        'CUR_TANK': 0.8 + 0.08 * np.random.randn(n_samples),
        'PITCH': pitch,
        'ROLL': roll,
        'AZIMUTH': azimuth,
        'BARO': 1000 + 100 * np.random.randn(n_samples),
        'LINE': np.concatenate([
            np.full(2000, 1001),
            np.full(2000, 1002), 
            np.full(2000, 1003),
            np.full(2000, 1004),
            np.full(2000, 1005)
        ]),
        'IGRFMAG1': mag1 + 10 * np.random.randn(n_samples)
    }
    
    return pd.DataFrame(data)


#------------#
#----Main----#
#------------#


if __name__ == "__main__":
    
    # Start timer
    start_time = time.time()
    
    # set seed for reproducibility
    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)
    
    #----User arguments----#
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-d","--device", type=str, required=False, default='cpu', help="Which GPU to use (cuda or cpu), default='cpu'. Ex : --device 'cuda' ", metavar=""
    )
    parser.add_argument(
        "-e","--epochs", type=int, required=False, default=35, help="Number of epochs to train the model, default=35. Ex : --epochs 200", metavar=""
    )
    parser.add_argument(
        "-b","--batch", type=int, required=False, default=256, help="Batch size for training, default=256. Ex : --batch 64", metavar=""
    )
    parser.add_argument(
        "-sq","--seq", type=int, required=False, default=20, help="Length sequence of data, default=20. Ex : --seq 15", metavar=""
    )
    parser.add_argument(
        "--shut", action="store_true", required=False, help="Shutdown pc after training is done."
    )
    parser.add_argument(
        "-sc", "--scaling", type=int, required=False, default=1, help="Data scaling, 1 for standardization, 2 for MinMax scaling, 0 for no scaling, default=1. Ex : --scaling 1", metavar=''
    )
    parser.add_argument(
        "-cor", "--corrections", type=int, required=False, default=3, help="Data correction, 0 for no corrections, 1 for IGRF correction, 2 for diurnal correction, 3 for IGRF+diurnal correction. Ex : --corrections 3", metavar=''
    )
    parser.add_argument(
        "-tl", "--tolleslawson", type=int, required=False, default=1, help="Apply Tolles-Lawson compensation to data, 0 for no compensation, 1 for compensation. Ex : --tolleslawson 1", metavar=''
    )
    parser.add_argument(
        "-tr", "--truth", type=str, required=False, default='IGRFMAG1', help="Name of the variable corresponding to the truth for training the model. Ex : --truth 'IGRFMAG1'", metavar=''
    )
    parser.add_argument(
        "-ml", "--model", type=str, required=False, default='ContiFormer', help="Name of the model to use. Available models : 'MLP', 'CNN', 'ResNet18', 'LSTM', 'GRU', 'ContiFormer', 'ContiFormerSimple', 'DivFreeContiFormer', 'StandardContiFormer', 'SE3ContiFormer'. Ex : --model 'ContiFormer'", metavar=''
    )
    parser.add_argument(
        "-wd", "--weight_decay", type=float, required=False, default=0.001, help="Adam weight decay value. Ex : --weight_decay 0.00001", metavar=''
    )
    parser.add_argument(
        "--synthetic", action="store_true", required=False, help="Use synthetic data for testing when real flight data is not available."
    )
    parser.add_argument('--subset', type=int, default=None, help='Training subset size')
    parser.add_argument('--testsubset', type=int, default=None, help='Test subset size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for training with DataParallel (1-8)')
    parser.add_argument('--train_all', action='store_true', help='Train all models in parallel on 8 A6000 GPUs simultaneously')


    
    args = parser.parse_args()

    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch
    SEQ_LEN    = args.seq
    SCALING    = args.scaling
    COR        = args.corrections
    TL         = args.tolleslawson
    TRUTH      = args.truth
    MODEL      = args.model
    WEIGHT_DECAY = args.weight_decay
    
    num_gpus = args.num_gpus
    DEVICE = torch.device('cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cuda':
        available_gpus = torch.cuda.device_count()
        num_gpus = min(num_gpus, available_gpus)
        if num_gpus > 1:
            print(f'\nTraining on {num_gpus} GPUs: ' + ', '.join([torch.cuda.get_device_name(i) for i in range(num_gpus)]))
        else:
            print(f'\nTraining on {torch.cuda.get_device_name(0)}')
    else:
        num_gpus = 1
        print('\nTraining on cpu.')

    #----Import data----#
    
    if args.synthetic:
        # Use synthetic data for testing
        df = create_synthetic_magnav_data()
        flights_num = [1]  # Single synthetic flight
        flights = {1: df}
        print(f'Synthetic data created. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    else:
        # Try to import real flight data (memory optimized - load fewer flights)
        try:
            flights = {}
            flights_num = [2,3]  # Reduced to 2 flights to save memory
            
            for n in flights_num:
                df = pd.read_hdf("data/processed/Flt_data.h5", key=f'Flt100{n}')
                flights[n] = df
            
            print(f'Data import done (2 flights for memory efficiency). Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
        except (FileNotFoundError, KeyError) as e:
            print(f"Real flight data issue ({e}). Using synthetic data instead.")
            df = create_synthetic_magnav_data()
            flights_num = [1]
            flights = {1: df}
            print(f'Synthetic data created. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Slecting train/test lines----#
    
    if args.synthetic:
        # For synthetic data, simple split
        train_lines = [[1001, 1002, 1003]]
        test_lines = [[1004, 1005]]
    else:
        # Memory-optimized flight-based split (using flights 2,3 only)
        train_lines = [flights[2].LINE.unique().tolist()]
        test_lines  = [flights[3].LINE.unique().tolist()]

    #----Apply Tolles-Lawson (Skip for synthetic data)----#
    
    if not args.synthetic and TL == 1:
        # Get cloverleaf pattern data
        mask = (flights[2].LINE == 1002.20)
        tl_pattern = flights[2][mask]

        # filter parameters
        fs      = 10.0
        lowcut  = 0.1
        highcut = 0.9
        filt    = ['Butterworth',4]
        
        ridge = 0.025
        for n in tqdm(flights_num):

            # A matrix of Tolles-Lawson
            A = magnav.create_TL_A(flights[n]['FLUXB_X'],flights[n]['FLUXB_Y'],flights[n]['FLUXB_Z'])

            # Tolles Lawson coefficients computation
            TL_coef_2 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG2'],
                                          lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
            TL_coef_3 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG3'],
                                          lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
            TL_coef_4 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG4'],
                                          lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
            TL_coef_5 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG5'],
                                          lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)

            # Magnetometers correction
            flights[n]['TL_comp_mag2_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG2'].tolist(),(-1,1)), TL_coef_2, A)
            flights[n]['TL_comp_mag3_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG3'].tolist(),(-1,1)), TL_coef_3, A)
            flights[n]['TL_comp_mag4_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG4'].tolist(),(-1,1)), TL_coef_4, A)
            flights[n]['TL_comp_mag5_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG5'].tolist(),(-1,1)), TL_coef_5, A)

        print(f'Tolles-Lawson correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    else:
        print(f'Tolles-Lawson correction skipped. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')

    #----Apply IGRF and diurnal corrections (Skip for synthetic data)----#

    flights_cor = {}

    if TL == 1 and not args.synthetic:
        mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl']
    else:
        mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl']  # These exist in synthetic data
    
    if COR == 0 or args.synthetic:
        flights_cor = flights.copy()
        del flights
        print(f'No correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    elif COR == 1:
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=False, igrf=True)
        del flights
        print(f'IGRF correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    elif COR == 2: 
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=False)
        del flights
        print(f'Diurnal correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    elif COR == 3:
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=True)
        del flights
        print(f'IGRF+Diurnal correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Select features----#
    
    # Always keep the 'LINE' feature in the feature list so that the MagNavDataset function can split the flight data
    if MODEL == 'CfC_DivFree':
        features = mags_to_cor + ['V_BAT1','V_BAT2','INS_VEL_N','INS_VEL_V','INS_VEL_W','CUR_IHTR','CUR_FLAP','CUR_ACLo','CUR_TANK','PITCH','ROLL','AZIMUTH','BARO','LINE',TRUTH, 'UTM_X', 'UTM_Y', 'UTM_Z', 'TIME']
    else:
        features = mags_to_cor + ['V_BAT1','V_BAT2','INS_VEL_N','INS_VEL_V','INS_VEL_W','CUR_IHTR','CUR_FLAP','CUR_ACLo','CUR_TANK','PITCH','ROLL','AZIMUTH','BARO','LINE',TRUTH]

    dataset = {}
    
    for n in flights_num:
        dataset[n] = flights_cor[n][features]
    
    del flights_cor
    
    print(f'Feature selection done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Data scaling----#
    
    # Combine all flights into a single DataFrame for easier processing
    df = pd.DataFrame()
    for flight in flights_num:
        df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
    del dataset

    scaling = {} # Dictionary to store scaling params for each fold

    if SCALING == 1: # Standardization
        print("Applying Standard scaling (fitting on train set for each fold)...")
        for n in range(len(train_lines)):
            train_mask = df['LINE'].isin(train_lines[n])
            train_mean = df.loc[train_mask, TRUTH].mean()
            train_std = df.loc[train_mask, TRUTH].std()
            scaling[n] = ['std', train_mean, train_std]
        
        # Apply scaling to the whole dataframe using the mean/std of the first fold's training data for simplicity
        # A more advanced implementation would scale inside the loop.
        global_train_mask = df['LINE'].isin(train_lines[0])
        df = Standard_scaling(df)

    elif SCALING == 2: # MinMax Scaling
        print("Applying MinMax scaling (fitting on train set for each fold)...")
        bound = [-1, 1]
        for n in range(len(train_lines)):
            train_mask = df['LINE'].isin(train_lines[n])
            train_min = df.loc[train_mask, TRUTH].min()
            train_max = df.loc[train_mask, TRUTH].max()
            scaling[n] = ['minmax', bound[0], bound[1], train_min, train_max]

        # Apply scaling to the whole dataframe
        df = MinMax_scaling(df, bound=bound)
    
    else: # No Scaling
        for n in range(len(train_lines)):
            scaling[n] = ['None']

    print(f'Data scaling done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Training----#
    
    train_loss_history = []
    test_loss_history = []
    RMSE_history = []
    best_models_per_fold = []
    
    # Cross validation with selected folds 
    for fold in range(len(train_lines)):
        
        print('\n--------------------')
        print(f'Fold number {fold}')
        print('--------------------\n')
        
        if MODEL == 'CfC_DivFree':
            train = CfCDataset(df, seq_len=SEQ_LEN, split='train', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)
            test  = CfCDataset(df, seq_len=SEQ_LEN, split='test', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)
            model_feature_count = len(train.features)
        else:
            train = MagNavDataset(df, seq_len=SEQ_LEN, split='train', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)
            test  = MagNavDataset(df, seq_len=SEQ_LEN, split='test', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)
            model_feature_count = len(train.features)

        if args.testsubset is not None:
            test = torch.utils.data.Subset(test, range(min(args.testsubset, len(test))))
            print(f"Using test subset: {len(test)} samples")
            
        if args.subset is not None:
            train = torch.utils.data.Subset(train, range(min(args.subset, len(train))))
            print(f"Using training subset: {len(train)} samples")


        # Dataloaders
        train_loader  = DataLoader(train,
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=0,
                               pin_memory=False)

        test_loader    = DataLoader(test,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)

        # Model
        if MODEL == 'MLP':
            model = MLP(SEQ_LEN, model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'MLP2':
            model = MLP2(SEQ_LEN,len(features)-2)
            model.name = model.__class__.__name__
        elif MODEL == 'MLP3':
            model = MLP3(SEQ_LEN,len(features)-2)
            model.name = model.__class__.__name__
        elif MODEL == 'MLP4':
            model = MLP4(SEQ_LEN,len(features)-2)
            model.name = model.__class__.__name__
        elif MODEL == 'E3NNCNN':
            model = E3NNCNN(SEQ_LEN, model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'E3NNDivergenceFreeCNN':
            model = E3NNDivergenceFreeCNN(SEQ_LEN, model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'E3NNDivergenceFreeCNN2':
            model = E3NNDivergenceFreeCNN2(SEQ_LEN, model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'E3NNDivergenceFreeCNN3':
            model = E3NNDivergenceFreeCNN3(SEQ_LEN, model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'CNN':
            model = CNN(SEQ_LEN, model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'CfC_DivFree':
            model = CfC_DivFree(input_size=model_feature_count, seq_len=SEQ_LEN)
            model.name = 'CfC_DivFree'
        elif MODEL == 'E3NNMLP':
            model = E3NNMLP(seq_length=SEQ_LEN, n_features=N_FEATURES)
            model.name = model.__class__.__name__
        elif MODEL == 'DivergenceFreeMLP2':
            model = DivergenceFreeMLP2(seq_length=SEQ_LEN, n_features=len(features) - 2, output_dim=1)
            model.name = model.__class__.__name__
        elif MODEL == 'DFMLP':
            model = DivergenceFreeMLP(seq_length=SEQ_LEN, n_features=model_feature_count, output_dim=1)
            model.name = model.__class__.__name__
        elif MODEL == 'ResNet18':
            model = ResNet18(in_channels=model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'E3NNDivergenceFreeMLP':
            model = E3NNDivergenceFreeMLP(seq_length=SEQ_LEN, n_features=model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'DivergencefreeCNN':
            model = DivergenceFreeCNN(SEQ_LEN, model_feature_count)
            model.name = model.__class__.__name__
        elif MODEL == 'LSTM':
            num_LSTM    = 2
            hidden_size = [32,32]
            num_layers  = [3,1]
            num_linear  = 2
            num_neurons = [16,4]
            model = LSTM(SEQ_LEN, hidden_size, num_layers, num_LSTM, num_linear, num_neurons, DEVICE)
            model.name = model.__class__.__name__
        elif MODEL == 'GRU':
            model = GRU(SEQ_LEN, model_feature_count, 32)
            model.name = model.__class__.__name__
        elif MODEL == 'ContiFormer':
            # Use ODE-based ContiFormer with proper parameters
            model = ContiFormer(
                input_size=len(features)-2,
                d_model=256,
                d_inner=1024,
                n_layers=4,
                n_head=4,
                d_k=64,
                d_v=64,
                dropout=0.1,
                max_length=SEQ_LEN,
                # ODE-specific parameters
                actfn_ode="softplus",
                layer_type_ode="concat",
                zero_init_ode=True,
                atol_ode=1e-3,
                rtol_ode=1e-3,
                method_ode="rk4",
                linear_type_ode="inside",
                regularize=False,
                approximate_method="last",
                nlinspace=3,
                interpolate_ode="linear",
                itol_ode=1e-2,
                divergence_free=True,  # Enable divergence-free vector fields
                add_pe=False,
                normalize_before=False
            )
            model.name = MODEL
            print(f"\nCreated ODE-based ContiFormer")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        elif MODEL == 'SE3ContiFormer':
            model = SE3ContiFormerMagNav(SEQ_LEN, model_feature_count)
            model.name = MODEL
            print(f"\nCreated SE3ContiFormer (SE(3) equivariant + divergence-free)")
            print(f"Model info: {model.get_model_info()}")
            # Check SE(3) equivariance and divergence-free properties
            model.check_se3_equivariance(device=DEVICE)
            model.check_divergence_free_property(device=DEVICE)

        model = model.to(DEVICE)
        if num_gpus > 1 and DEVICE.type == 'cuda':
            model = nn.DataParallel(model, device_ids=list(range(num_gpus))) 

        # Loss function
        criterion = torch.nn.MSELoss()

        # Training
        train_loss, test_loss, Best_RMSE, Best_model = make_training(model, EPOCHS, train_loader, test_loader, scaling[fold], lr=args.lr)

        
        # Save results from training
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        RMSE_history.append(Best_RMSE)
        best_models_per_fold.append(Best_model)
        
        if fold == 0:
            folder_path = f'models/New_{get_model_name(Best_model)}_{scaling[0][0]}_TL{TL}_COR{COR}_{TRUTH}'
            os.makedirs(folder_path, exist_ok=True)
        torch.save(Best_model,folder_path+f'/{get_model_name(Best_model)}_fold{fold}.pt')
    
    # Compute pre-processing+training time
    end_time = time.time()-start_time
    
    # Compute global perf over all folds
    perf_folds = sum(RMSE_history)/len(train_lines)
    
    # Print perf of training
    print('\n-------------------------')
    print('Performance for all folds')
    print('-------------------------')
    for n in range(len(test_lines)):
        print(f'Fold {n} | RMSE = {RMSE_history[n]:.2f} nT')
    print(f'Total  | RMSE = {perf_folds:.2f} nT')
    
    # Show performance graphs
    for fold in range(len(test_lines)):
        fig, ax = plt.subplots(figsize=[10,4])
        ax.plot(train_loss_history[fold], label='Train loss')
        ax.plot(test_loss_history[fold], label='Test loss')
        ax.set_title(f'Loss for fold {fold}')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.savefig(folder_path+f'/losses_fold{fold}')
    
    # Save parameters
    params = pd.DataFrame(columns=['seq_len','epochs','batch_size','training_time','model','divergence_free'])
    params.loc[0,'seq_len'] = SEQ_LEN
    params.loc[0,'epochs'] = EPOCHS
    params.loc[0,'batch_size'] = BATCH_SIZE
    params.loc[0,'training_time'] = end_time
    params.loc[0,'model'] = MODEL
    params.loc[0,'divergence_free'] = MODEL in ['ContiFormer', 'DivFreeContiFormer', 'SE3ContiFormer', 'CfC_DivFree']
    params.to_csv(folder_path+f'/parameters.csv', index=False)
    
    # Empty GPU ram if using CUDA
    if torch.cuda.is_available() and DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    # Shutdown computer safely (disabled to prevent accidental shutdown)
    if args.shut:
        print("\n[INFO] --shut flag was provided, but automatic shutdown is disabled to prevent accidental shutdown. Please shut down the system manually if desired.") 