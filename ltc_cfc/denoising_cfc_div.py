import sys
sys.path.append('CfC')



# ────────────────────────────────────────────────────────────────────
# IMPORTS
# ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
from torch_cfc import Cfc
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch.utils.data import Dataset, DataLoader, random_split



# ────────────────────────────────────────────────────────────────────
# GLOBALS
# ────────────────────────────────────────────────────────────────────
FS = 100 # sensor sample-rate [Hz]
CAL_DURATION = 120 # seconds to use for TL calibration



# ────────────────────────────────────────────────────────────────────
# PRE-PROCESSING HELPERS (TL → IGRF → DIURNAL)
# ────────────────────────────────────────────────────────────────────
def create_TL_A(Bx, By, Bz, Bt_scale=50_000, fs=FS):
    Bt = np.sqrt(Bx**2 + By**2 + Bz**2)
    s  = Bt / Bt_scale
    cosX, cosY, cosZ = Bx/Bt, By/Bt, Bz/Bt
    dcosX = np.gradient(cosX, 1/fs)
    dcosY = np.gradient(cosY, 1/fs)
    dcosZ = np.gradient(cosZ, 1/fs)
    A_perm = np.column_stack((cosX, cosY, cosZ))
    A_ind  = np.column_stack((s*cosX*cosX, s*cosX*cosY, s*cosX*cosZ,
                              s*cosY*cosY, s*cosY*cosZ, s*cosZ*cosZ))
    A_eddy = np.column_stack((s*cosX*dcosX, s*cosX*dcosY, s*cosX*dcosZ,
                              s*cosY*dcosX, s*cosY*dcosY, s*cosY*dcosZ,
                              s*cosZ*dcosX, s*cosZ*dcosY, s*cosZ*dcosZ))
    return np.column_stack((A_perm, A_ind, A_eddy))

def butter_band(data, fs, low=0.1, high=0.9, order=2):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, data)

def butter_low(data, fs, cutoff_hz, order=2):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff_hz/nyq, btype='low')
    return signal.filtfilt(b, a, data)

def fit_tl(Bx, By, Bz, mag_scalar, fs=FS, low=0.1, high=0.9,
           trim=20, ridge=1.0):
    A  = create_TL_A(Bx, By, Bz, fs=fs)
    Af = np.column_stack([butter_band(A[:, i], fs, low, high)
                          for i in range(A.shape[1])])
    mf = butter_band(mag_scalar, fs, low, high)
    A_t, m_t = Af[trim:-trim], mf[trim:-trim, None]
    coef = np.linalg.solve(A_t.T @ A_t + ridge*np.eye(A_t.shape[1]),
                           A_t.T @ m_t)
    return coef.squeeze()

def predict_tl(Bx, By, Bz, coef, fs=FS):
    return create_TL_A(Bx, By, Bz, fs=fs) @ coef



# ────────────────────────────────────────────────────────────────────
# LOADING FLIGHT DATA & APPLY CORRECTIONS
# ────────────────────────────────────────────────────────────────────
flight_2 = pd.read_csv('data/processed/Flt1002.csv')
flight_2.rename(columns={'Time [s]': 'TIME'}, inplace=True)

Bx = flight_2['FLUXB_X'].values
By = flight_2['FLUXB_Y'].values
Bz = flight_2['FLUXB_Z'].values
mag_raw = np.sqrt(Bx**2 + By**2 + Bz**2)

cal_mask = flight_2['TIME'] < CAL_DURATION
coef = fit_tl(Bx[cal_mask], By[cal_mask], Bz[cal_mask],
              mag_raw[cal_mask])

p_vehicle = predict_tl(Bx, By, Bz, coef)
mag_tl = mag_raw - p_vehicle + p_vehicle.mean()
flight_2['MAG_TL'] = mag_tl

mag_noigrf = mag_tl - flight_2['IGRFMAG1']
flight_2['MAG_noIGRF'] = mag_noigrf

drift = butter_low(mag_noigrf, FS, cutoff_hz=1/7200)
flight_2['MAG_FINAL'] = mag_noigrf - drift

# creating corrected vector target
mag_raw_vec = flight_2[['FLUXB_X','FLUXB_Y','FLUXB_Z']].values
mag_raw_vec_norm = mag_raw_vec / np.linalg.norm(mag_raw_vec, axis=1, keepdims=True)
mag_corrected_vec = mag_raw_vec_norm * flight_2['MAG_FINAL'].values[:, None]
flight_2['FLUXB_X_CORR'] = mag_corrected_vec[:, 0]
flight_2['FLUXB_Y_CORR'] = mag_corrected_vec[:, 1]
flight_2['FLUXB_Z_CORR'] = mag_corrected_vec[:, 2]



# ────────────────────────────────────────────────────────────────────
# DEFINING DATASET CLASS
# ────────────────────────────────────────────────────────────────────
class CfCMagNavVectorDataset(Dataset):
    def __init__(self, df, sequence_length=100):
        self.sequence_length = sequence_length

        self.features = [
            'UNCOMPMAG1','UNCOMPMAG2','UNCOMPMAG3','UNCOMPMAG4','UNCOMPMAG5',
            'PITCH','ROLL','AZIMUTH','BARO','TRUE_AS',
            'INS_VEL_N','INS_VEL_V','INS_VEL_W',
            'CUR_ACLo','CUR_FLAP','CUR_TANK','CUR_IHTR','V_BAT1','V_BAT2'
        ]
        self.vector_cols = ['FLUXB_X_CORR','FLUXB_Y_CORR','FLUXB_Z_CORR']
        self.scalar_col  = 'MAG_FINAL'
        self.coord_cols  = ['UTM_X','UTM_Y','UTM_Z']

        self.X        = torch.tensor(df[self.features].values, dtype=torch.float32)
        self.y_vector = torch.tensor(df[self.vector_cols].values, dtype=torch.float32)
        self.y_scalar = torch.tensor(df[self.scalar_col].values,  dtype=torch.float32)
        self.coords_t = torch.tensor(df[self.coord_cols].values, dtype=torch.float32)
        self.t        = torch.tensor(df['TIME'].values,           dtype=torch.float32)

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.sequence_length)
        delta_t = torch.cat([torch.zeros(1),
                             self.t[sl][1:] - self.t[sl][:-1]])
        return (self.X[sl],
                delta_t.unsqueeze(1),
                self.y_scalar[sl].unsqueeze(1),
                self.y_vector[sl],
                self.coords_t[sl])



# ────────────────────────────────────────────────────────────────────
# DEFINING MODEL CLASS
# ────────────────────────────────────────────────────────────────────
class CfCMagNavDivergenceFreeModel(LightningModule):
    def __init__(self, input_size, hidden_size=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.cfc = Cfc(in_features=input_size,
                       hidden_size=hidden_size,
                       out_feature=3,
                       hparams=dict(backbone_activation='silu',
                                    backbone_units=hidden_size,
                                    backbone_layers=2,
                                    init=1.0,
                                    minimal=False),
                       return_sequences=True,
                       use_mixed=True,
                       use_ltc=True)
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def curl(field, coords):

        Ax, Ay, Az = field[..., 0], field[..., 1], field[..., 2]
        X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]


        # numerators
        dAx = Ax[:, 2:] - Ax[:, :-2]
        dAy = Ay[:, 2:] - Ay[:, :-2]
        dAz = Az[:, 2:] - Az[:, :-2]
        # denominators
        dX = X[:, 2:] - X[:, :-2]
        dY = Y[:, 2:] - Y[:, :-2]
        dZ = Z[:, 2:] - Z[:, :-2]

        eps = 1e-8

        dAz_dY = dAz / (dY + eps)
        dAy_dZ = dAy / (dZ + eps)
        dAx_dZ = dAx / (dZ + eps)
        dAz_dX = dAz / (dX + eps)
        dAy_dX = dAy / (dX + eps)
        dAx_dY = dAx / (dY + eps)

        Bx = dAz_dY - dAy_dZ
        By = dAx_dZ - dAz_dX
        Bz = dAy_dX - dAx_dY

        B = torch.stack([Bx, By, Bz], dim=2)
        return torch.cat([B[:, :1], B, B[:, -1:]], dim=1)

    def forward(self, x, delta_t, coords):
        A = self.cfc(x, timespans=delta_t)
        B = self.curl(A, coords)
        B_mag = torch.sqrt((B**2).sum(2, keepdim=True))
        return B, B_mag

    def training_step(self, batch, batch_idx):
        x, dt, y_s, y_v, coords = batch
        B_pred, B_mag_pred = self(x, dt, coords)
        loss = self.loss_fn(B_pred, y_v) + self.loss_fn(B_mag_pred, y_s)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, dt, y_s, y_v, coords = batch
        B_pred, B_mag_pred = self(x, dt, coords)
        vloss = self.loss_fn(B_pred, y_v) + self.loss_fn(B_mag_pred, y_s)
        self.log('val_loss', vloss, prog_bar=True)
        return vloss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



# ────────────────────────────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    dataset = CfCMagNavVectorDataset(flight_2)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # normalizing data after splitting - calculating mean/std from the training set only
    train_X_cat = torch.cat([train_ds.dataset.X[i] for i in train_ds.indices], dim=0)
    train_y_vec_cat = torch.cat([train_ds.dataset.y_vector[i] for i in train_ds.indices], dim=0)
    train_y_sca_cat = torch.cat([train_ds.dataset.y_scalar[i] for i in train_ds.indices], dim=0)
    train_coords_cat = torch.cat([train_ds.dataset.coords_t[i] for i in train_ds.indices], dim=0)

    X_mean, X_std = train_X_cat.mean(0), train_X_cat.std(0)
    y_vec_mean, y_vec_std = train_y_vec_cat.mean(0), train_y_vec_cat.std(0)
    y_sca_mean, y_sca_std = train_y_sca_cat.mean(), train_y_sca_cat.std()
    coords_mean, coords_std = train_coords_cat.mean(0), train_coords_cat.std()

    # applying the same transformation to the whole dataset
    dataset.X = (dataset.X - X_mean) / (X_std + 1e-8)
    dataset.y_vector = (dataset.y_vector - y_vec_mean) / (y_vec_std + 1e-8)
    dataset.y_scalar = (dataset.y_scalar - y_sca_mean) / (y_sca_std + 1e-8)
    dataset.coords_t = (dataset.coords_t - coords_mean) / (coords_std + 1e-8)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)

    model = CfCMagNavDivergenceFreeModel(input_size=len(dataset.features))

    trainer = Trainer(max_epochs=30,
                      precision='16-mixed',
                      accelerator='gpu',
                      devices=1,
                      accumulate_grad_batches=2,
                      gradient_clip_val=1.0,
                      callbacks=[RichProgressBar(),
                                 ModelCheckpoint(monitor='val_loss',
                                                 save_top_k=1,
                                                 mode='min')])

    trainer.fit(model, train_loader, val_loader)