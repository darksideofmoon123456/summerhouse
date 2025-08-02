import sys
sys.path.append('CfC')                  # CfC repo path



# ────────────────────────────────────────────────────────────────────
# IMPORTS
# ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
from torch_cfc import Cfc
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch.utils.data import Dataset, DataLoader, random_split



# ────────────────────────────────────────────────────────────────────
# GLOBALS
# ────────────────────────────────────────────────────────────────────
FS            = 100      # Hz
CAL_DURATION  = 120      # s



# ────────────────────────────────────────────────────────────────────
# FILE LOADER
# ────────────────────────────────────────────────────────────────────
def load_flight(path: str) -> pd.DataFrame:
    suf = path.split('.')[-1].lower()
    if suf == 'csv':
        df = pd.read_csv(path)
    elif suf in ('h5', 'hdf5'):
        try:
            df = pd.read_hdf(path)
        except (ValueError, KeyError):
            import h5py
            with h5py.File(path) as f:
                def collect(node, pre=''):
                    out = {}
                    for k, v in node.items():
                        key = f'{pre}{k}'
                        if isinstance(v, h5py.Dataset) and v.shape != ():
                            out[key] = v[()]
                        elif isinstance(v, h5py.Group):
                            out.update(collect(v, key + '/'))
                    return out
                raw = collect(f)
            df = pd.DataFrame({k: np.asarray(v).ravel() for k, v in raw.items()})
    else:
        raise ValueError('unsupported file type')
    df.rename(columns=lambda c: c.strip(), inplace=True)
    if 'Time [s]' in df and 'TIME' not in df:
        df.rename(columns={'Time [s]': 'TIME'}, inplace=True)
    return df



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

def butter_band(x, fs, low=0.1, high=0.9, order=2):
    nyq = 0.5*fs
    b, a = signal.butter(order, [low/nyq, high/nyq], 'band')
    return signal.filtfilt(b, a, x)

def butter_low(x, fs, cutoff_hz, order=2):
    nyq = 0.5*fs
    b, a = signal.butter(order, cutoff_hz/nyq, 'low')
    return signal.filtfilt(b, a, x)

def fit_tl(Bx, By, Bz, m, fs=FS, low=0.1, high=0.9, trim=20, ridge=1.0):
    A  = create_TL_A(Bx, By, Bz, fs=fs)
    Af = np.column_stack([butter_band(A[:, i], fs, low, high) for i in range(A.shape[1])])
    mf = butter_band(m, fs, low, high)
    A_t, m_t = Af[trim:-trim], mf[trim:-trim,None]
    coef = np.linalg.solve(A_t.T@A_t + ridge*np.eye(A_t.shape[1]), A_t.T@m_t)
    return coef.squeeze()

def predict_tl(Bx, By, Bz, coef, fs=FS):
    return create_TL_A(Bx, By, Bz, fs=fs) @ coef



# ────────────────────────────────────────────────────────────────────
# LOAD & CORRECT DATA
# ────────────────────────────────────────────────────────────────────
flight_2 = load_flight('data/processed/Flt1002.h5')

Bx, By, Bz = flight_2[['FLUXB_X','FLUXB_Y','FLUXB_Z']].values.T
mag_raw    = np.sqrt(Bx**2 + By**2 + Bz**2)

coef       = fit_tl(Bx[flight_2['TIME']<CAL_DURATION],
                    By[flight_2['TIME']<CAL_DURATION],
                    Bz[flight_2['TIME']<CAL_DURATION],
                    mag_raw[flight_2['TIME']<CAL_DURATION])

p_vehicle  = predict_tl(Bx, By, Bz, coef)
mag_tl     = mag_raw - p_vehicle + p_vehicle.mean()
flight_2['MAG_TL'] = mag_tl

mag_noigrf = mag_tl - flight_2['IGRFMAG1']
flight_2['MAG_noIGRF'] = mag_noigrf
flight_2['MAG_FINAL']  = mag_noigrf - butter_low(mag_noigrf, FS, 1/7200)

vec = flight_2[['FLUXB_X','FLUXB_Y','FLUXB_Z']].values
vec_norm = vec / np.linalg.norm(vec, axis=1, keepdims=True)
corr_vec = vec_norm * flight_2['MAG_FINAL'].values[:,None]
flight_2[['FLUXB_X_CORR','FLUXB_Y_CORR','FLUXB_Z_CORR']] = corr_vec



# ────────────────────────────────────────────────────────────────────
# DATASET
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

    def __len__(self):  return len(self.X)-self.sequence_length+1

    def __getitem__(self, idx):
        sl = slice(idx, idx+self.sequence_length)
        delta_t = torch.cat([torch.zeros(1), self.t[sl][1:]-self.t[sl][:-1]])
        return (self.X[sl],
                delta_t.unsqueeze(1),
                self.y_scalar[sl].unsqueeze(1),
                self.y_vector[sl],
                self.coords_t[sl])




# ────────────────────────────────────────────────────────────────────
# MODEL   (CfC + LTC  → vector potential  → curl ⇒ div-free B)
# ────────────────────────────────────────────────────────────────────
class CfCMagNavDivFree(LightningModule):
    def __init__(self, input_size, hidden_size=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.cfc = Cfc(in_features=input_size,
                       hidden_size=hidden_size,
                       out_feature=3,                    # vector potential A
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
    def curl(A, coords):
        Ax, Ay, Az = A[...,0], A[...,1], A[...,2]
        X,  Y,  Z  = coords[...,0], coords[...,1], coords[...,2]

        eps = 1e-8
        dAx = Ax[:,2:]-Ax[:,:-2]; dAy = Ay[:,2:]-Ay[:,:-2]; dAz = Az[:,2:]-Az[:,:-2]
        dX  = X[:,2:]-X[:,:-2];   dY  = Y[:,2:]-Y[:,:-2];   dZ  = Z[:,2:]-Z[:,:-2]

        dAz_dY = dAz/(dY+eps); dAy_dZ = dAy/(dZ+eps)
        dAx_dZ = dAx/(dZ+eps); dAz_dX = dAz/(dX+eps)
        dAy_dX = dAy/(dX+eps); dAx_dY = dAx/(dY+eps)

        Bx = dAz_dY-dAy_dZ
        By = dAx_dZ-dAz_dX
        Bz = dAy_dX-dAx_dY
        B  = torch.stack((Bx,By,Bz),2)
        return torch.cat([B[:,:1], B, B[:,-1:]],1)   # restoring length

    def forward(self, x, dt, coords):
        A      = self.cfc(x, timespans=dt)           # vector potential
        B      = self.curl(A, coords)                # divergence-free B
        B_mag  = B.norm(dim=2, keepdim=True)
        return B, B_mag

    def _shared_step(self, batch):
        x, dt, y_s, y_v, coords = batch
        B_pred, B_mag_pred = self(x, dt, coords)
        return self.loss_fn(B_pred, y_v)+self.loss_fn(B_mag_pred, y_s)

    def training_step  (self,b,i): loss=self._shared_step(b); self.log('train_loss',loss); return loss
    def validation_step(self,b,i): v=self._shared_step(b);   self.log('val_loss',v);     return v
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)




# ────────────────────────────────────────────────────────────────────
# TRAIN (5-FOLD CROSS-VALIDATION)
# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from sklearn.model_selection import KFold

    NUM_FOLDS = 5
    full_df   = flight_2                       # pre-processed DataFrame from earlier
    kf        = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(full_df))), 1):
        print(f'\n── Fold {fold}/{NUM_FOLDS} ───────────────────────────')

        ds = CfCMagNavVectorDataset(full_df, sequence_length=100)
        tr_ds = torch.utils.data.Subset(ds, tr_idx)
        va_ds = torch.utils.data.Subset(ds, va_idx)

        # normalising using training subset only
        idxs = tr_idx
        X_mu,  X_sd  = torch.cat([ds.X[i]        for i in idxs]).mean(0), torch.cat([ds.X[i]        for i in idxs]).std(0)
        yv_mu,yv_sd  = torch.cat([ds.y_vector[i] for i in idxs]).mean(0), torch.cat([ds.y_vector[i] for i in idxs]).std(0)
        ys_mu,ys_sd  = torch.cat([ds.y_scalar[i] for i in idxs]).mean(),   torch.cat([ds.y_scalar[i] for i in idxs]).std()
        co_mu,co_sd  = torch.cat([ds.coords_t[i] for i in idxs]).mean(0), torch.cat([ds.coords_t[i] for i in idxs]).std(0)

        ds.X        = (ds.X        - X_mu)/(X_sd +1e-8)
        ds.y_vector = (ds.y_vector - yv_mu)/(yv_sd+1e-8)
        ds.y_scalar = (ds.y_scalar - ys_mu)/(ys_sd+1e-8)
        ds.coords_t = (ds.coords_t - co_mu)/(co_sd+1e-8)

        # loaders
        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=64)

        # model & trainer
        model = CfCMagNavDivFree(input_size=len(ds.features))

        trainer = Trainer(max_epochs=30,
                          precision='16-mixed',
                          accelerator='gpu',
                          devices=1,
                          callbacks=[
                              RichProgressBar(),
                              ModelCheckpoint(dirpath=f'ckpt/fold{fold}',
                                              filename='best',
                                              monitor='val_loss',
                                              mode='min')
                          ])

        trainer.fit(model, tr_loader, va_loader)