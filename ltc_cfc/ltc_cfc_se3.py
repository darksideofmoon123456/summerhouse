import sys
sys.path.append('CfC')                 # clone: https://github.com/raminmh/CfC.git



# ────────────────────────────────────────────────────────────────────
# IMPORTS
# ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch.utils.data import Dataset, DataLoader, random_split
from e3nn import o3
from e3nn.nn import Gate



# ────────────────────────────────────────────────────────────────────
# GLOBALS
# ────────────────────────────────────────────────────────────────────
FS = 100          # sensor sample-rate [Hz]
CAL_DURATION = 120  # seconds to use for TL calibration



# ────────────────────────────────────────────────────────────────────
# DATA-LOADING HELPER
# ────────────────────────────────────────────────────────────────────
def load_flight(path: str) -> pd.DataFrame:
    suffix = path.split('.')[-1].lower()
    if suffix == "csv":
        df = pd.read_csv(path)
    elif suffix in ("h5", "hdf5"):
        try:
            df = pd.read_hdf(path)
        except (ValueError, KeyError):
            import h5py
            with h5py.File(path, "r") as f:
                def collect(node, pre=""):
                    out = {}
                    for k, v in node.items():
                        key = f"{pre}{k}"
                        if isinstance(v, h5py.Dataset) and v.shape != ():
                            out[key] = v[()]
                        elif isinstance(v, h5py.Group):
                            out.update(collect(v, key + "/"))
                    return out
                raw = collect(f)
            df = pd.DataFrame({k: np.asarray(v).ravel() for k, v in raw.items()})
    else:
        raise ValueError("unsupported file type")

    df.rename(columns=lambda c: c.strip(), inplace=True)
    if "Time [s]" in df and "TIME" not in df:
        df.rename(columns={"Time [s]": "TIME"}, inplace=True)
    return df



# ────────────────────────────────────────────────────────────────────
# PRE-PROCESSING HELPERS  (TL → IGRF → DIURNAL)
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
    A = create_TL_A(Bx, By, Bz, fs=fs)
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
# LOADING & CORRECTING FLIGHT DATA
# ────────────────────────────────────────────────────────────────────
flight_2 = load_flight('data/processed/Flt1002.h5')

Bx = flight_2['FLUXB_X'].values
By = flight_2['FLUXB_Y'].values
Bz = flight_2['FLUXB_Z'].values
mag_raw = np.sqrt(Bx**2 + By**2 + Bz**2)

cal_mask = flight_2['TIME'] < CAL_DURATION
coef = fit_tl(Bx[cal_mask], By[cal_mask], Bz[cal_mask], mag_raw[cal_mask])

p_vehicle = predict_tl(Bx, By, Bz, coef)
mag_tl = mag_raw - p_vehicle + p_vehicle.mean()
flight_2['MAG_TL'] = mag_tl

mag_noigrf = mag_tl - flight_2['IGRFMAG1']
flight_2['MAG_noIGRF'] = mag_noigrf

drift = butter_low(mag_noigrf, FS, cutoff_hz=1/7200)
flight_2['MAG_FINAL'] = mag_noigrf - drift

mag_raw_vec = flight_2[['FLUXB_X','FLUXB_Y','FLUXB_Z']].values
mag_raw_vec_norm = mag_raw_vec / np.linalg.norm(mag_raw_vec, axis=1, keepdims=True)
mag_corrected_vec = mag_raw_vec_norm * flight_2['MAG_FINAL'].values[:, None]
flight_2['FLUXB_X_CORR'] = mag_corrected_vec[:, 0]
flight_2['FLUXB_Y_CORR'] = mag_corrected_vec[:, 1]
flight_2['FLUXB_Z_CORR'] = mag_corrected_vec[:, 2]



# ────────────────────────────────────────────────────────────────────
# DATASET
# ────────────────────────────────────────────────────────────────────
class CfCMagNavVectorDataset(Dataset):
    def __init__(self, df, sequence_length=100):
        self.sequence_length = sequence_length
        self.scalar_features = [
            'UNCOMPMAG1','UNCOMPMAG2','UNCOMPMAG3','UNCOMPMAG4','UNCOMPMAG5',
            'PITCH','ROLL','AZIMUTH','BARO','TRUE_AS',
            'CUR_ACLo','CUR_FLAP','CUR_TANK','CUR_IHTR','V_BAT1','V_BAT2'
        ]
        self.vector_features = ['INS_VEL_N','INS_VEL_V','INS_VEL_W']      # 3-D vector
        self.vector_cols = ['FLUXB_X_CORR','FLUXB_Y_CORR','FLUXB_Z_CORR']
        self.scalar_col  = 'MAG_FINAL'

        self.X_scalars = torch.tensor(df[self.scalar_features].values, dtype=torch.float32)
        self.X_vectors = torch.tensor(df[self.vector_features].values, dtype=torch.float32)
        self.y_vector  = torch.tensor(df[self.vector_cols].values, dtype=torch.float32)
        self.y_scalar  = torch.tensor(df[self.scalar_col].values,  dtype=torch.float32)

    def __len__(self):
        return len(self.X_scalars) - self.sequence_length + 1

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.sequence_length)
        return (self.X_scalars[sl],
                self.X_vectors[sl],
                self.y_scalar[sl].unsqueeze(1),
                self.y_vector[sl])



# ────────────────────────────────────────────────────────────────────
# MODEL
# ────────────────────────────────────────────────────────────────────
class SE3_CfC_Model(LightningModule):
    def __init__(self, num_scalar_features, num_vector_features, hidden_irreps="16x0e + 8x1o", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.input_irreps  = o3.Irreps(f"{num_scalar_features}x0e + {num_vector_features}x1o")
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_irreps = o3.Irreps("1x1o")        # predicting B vector

        self.rnn_cell = Gate(self.input_irreps + self.hidden_irreps,
                             [o3.torch.tanh, torch.sigmoid],
                             self.hidden_irreps)
        self.output_head = o3.Linear(self.hidden_irreps, self.output_irreps)
        self.loss_fn = nn.MSELoss()

    def forward(self, x_scalars, x_vectors):
        batch_size, seq_len, _ = x_scalars.shape
        hidden_state = self.hidden_irreps.zeros(batch_size, device=self.device)

        for t in range(seq_len):
            scalars_t = x_scalars[:, t, :]            # (B, n_scalar)
            vectors_t = x_vectors[:, t, :]            # (B, 3)
            input_t   = torch.cat([scalars_t, vectors_t], dim=1)
            hidden_state = self.rnn_cell(torch.cat([input_t, hidden_state], dim=1))

        B_pred     = self.output_head(hidden_state)          # (B, 3)
        B_mag_pred = B_pred.norm(dim=1, keepdim=True)        # (B, 1)
        return B_pred, B_mag_pred

    def training_step(self, batch, batch_idx):
        x_s, x_v, y_s, y_v = batch
        y_s_last, y_v_last = y_s[:, -1], y_v[:, -1]
        B_pred, B_mag_pred = self(x_s, x_v)
        loss = self.loss_fn(B_pred, y_v_last) + self.loss_fn(B_mag_pred, y_s_last)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_s, x_v, y_s, y_v = batch
        y_s_last, y_v_last = y_s[:, -1], y_v[:, -1]
        B_pred, B_mag_pred = self(x_s, x_v)
        val_loss = self.loss_fn(B_pred, y_v_last) + self.loss_fn(B_mag_pred, y_s_last)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



# ────────────────────────────────────────────────────────────────────
# TRAINING (5-FOLD CROSS-VALIDATION)
# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    NUM_FOLDS = 5
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(flight_2))), 1):
        print(f'\n── Fold {fold}/{NUM_FOLDS} ───────────────────────────')

        # fresh dataset so each fold starts un-normalised
        ds = CfCMagNavVectorDataset(flight_2, sequence_length=20)
        tr_ds = torch.utils.data.Subset(ds, tr_idx)
        va_ds = torch.utils.data.Subset(ds, va_idx)

        # computing μ/σ on training subset only
        def cat(tensor, idx): return torch.cat([tensor[i:i+ds.sequence_length] for i in idx], 0)
        Xs_mu, Xs_sd = cat(ds.X_scalars, tr_idx).mean(0), cat(ds.X_scalars, tr_idx).std(0)
        Xv_mu, Xv_sd = cat(ds.X_vectors, tr_idx).mean(0), cat(ds.X_vectors, tr_idx).std(0)
        yv_mu, yv_sd = cat(ds.y_vector , tr_idx).mean(0), cat(ds.y_vector , tr_idx).std(0)
        ys_mu, ys_sd = cat(ds.y_scalar , tr_idx).mean() , cat(ds.y_scalar , tr_idx).std()

        ds.X_scalars = (ds.X_scalars - Xs_mu) / (Xs_sd + 1e-8)
        ds.X_vectors = (ds.X_vectors - Xv_mu) / (Xv_sd + 1e-8)
        ds.y_vector  = (ds.y_vector  - yv_mu) / (yv_sd + 1e-8)
        ds.y_scalar  = (ds.y_scalar  - ys_mu) / (ys_sd + 1e-8)

        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=4)
        va_loader = DataLoader(va_ds, batch_size=32, num_workers=4)

        model = SE3_CfC_Model(len(ds.scalar_features),
                              len(ds.vector_features)//3)

        trainer = Trainer(max_epochs=50,
                          precision='16-mixed',
                          accelerator='gpu',
                          devices=1,
                          accumulate_grad_batches=2,
                          gradient_clip_val=1.0,
                          callbacks=[RichProgressBar(),
                                     ModelCheckpoint(dirpath=f'ckpt/fold{fold}',
                                                     filename='best',
                                                     monitor='val_loss',
                                                     mode='min')])

        trainer.fit(model, tr_loader, va_loader)