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
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch.utils.data import Dataset, DataLoader, random_split
from e3nn import o3



# ────────────────────────────────────────────────────────────────────
# GLOBALS
# ────────────────────────────────────────────────────────────────────
FS = 100 # sensor sample-rate [Hz]
CAL_DURATION = 120 # seconds to use for TL calibration



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
                def _gather(node, prefix=""):
                    out = {}
                    for k, v in node.items():
                        key = f"{prefix}{k}"
                        if isinstance(v, h5py.Dataset) and v.shape != ():
                            out[key] = v[()]
                        elif isinstance(v, h5py.Group):
                            out.update(_gather(v, key + "/"))
                    return out

                raw = _gather(f)
            # making everything 1-D so pandas can ingest it
            df = pd.DataFrame({k: np.asarray(v).ravel() for k, v in raw.items()})

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # unifying the time column name used throughout the pipeline
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
flight_2 = load_flight('data/processed/Flt1002.h5')
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
class CombinedDataset(Dataset):
    def __init__(self, df, sequence_length=100):
        self.sequence_length = sequence_length

        self.scalar_features = [
            'UNCOMPMAG1','UNCOMPMAG2','UNCOMPMAG3','UNCOMPMAG4','UNCOMPMAG5',
            'PITCH','ROLL','AZIMUTH','BARO','TRUE_AS',
            'CUR_ACLo','CUR_FLAP','CUR_TANK','CUR_IHTR','V_BAT1','V_BAT2'
        ]
        self.vector_features = ['INS_VEL_N','INS_VEL_V','INS_VEL_W']
        self.vector_cols = ['FLUXB_X_CORR','FLUXB_Y_CORR','FLUXB_Z_CORR']
        self.scalar_col  = 'MAG_FINAL'
        self.coord_cols  = ['UTM_X','UTM_Y','UTM_Z']

        self.X_scalars = torch.tensor(df[self.scalar_features].values, dtype=torch.float32)
        self.X_vectors = torch.tensor(df[self.vector_features].values, dtype=torch.float32)
        self.y_vector = torch.tensor(df[self.vector_cols].values, dtype=torch.float32)
        self.y_scalar = torch.tensor(df[self.scalar_col].values,  dtype=torch.float32)
        self.coords_t = torch.tensor(df[self.coord_cols].values, dtype=torch.float32)
        self.t        = torch.tensor(df['TIME'].values,           dtype=torch.float32)

    def __len__(self):
        return len(self.X_scalars) - self.sequence_length + 1

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.sequence_length)
        delta_t = torch.cat([torch.zeros(1), self.t[sl][1:] - self.t[sl][:-1]])
        
        return (self.X_scalars[sl],
                self.X_vectors[sl],
                delta_t.unsqueeze(1),
                self.y_scalar[sl].unsqueeze(1),
                self.y_vector[sl],
                self.coords_t[sl])



# ────────────────────────────────────────────────────────────────────
# DEFINING MODEL CLASS
# ────────────────────────────────────────────────────────────────────
class SE3_CfC_DivFree_Model(LightningModule):
    def __init__(self, num_scalar_features, num_vector_features,
                 hidden_dim=64, hidden_irreps="16x0e + 8x1o", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # ---------- CfC with LTC ----------------------------------- ▲
        self.cfc = Cfc(
            in_features=num_scalar_features + num_vector_features * 3,
            hidden_size=hidden_dim,
            out_feature=hidden_dim,         # keep size constant
            hparams={
                "backbone_units": hidden_dim,
                "backbone_layers": 2,
                "backbone_activation": "silu",
                "minimal": True
            },
            return_sequences=True,
            use_ltc=True,        # << LTC enabled
        )
        # ------------------------------------------------------------

        # mapping CfC hidden scalars → equivariant tensor
        self.scalar_irreps  = o3.Irreps(f"{hidden_dim}x0e")   # treating all h as scalars
        self.hidden_irreps  = o3.Irreps(hidden_irreps)
        self.to_hidden      = o3.Linear(self.scalar_irreps, self.hidden_irreps)
        self.output_irreps  = o3.Irreps("1x1o")               # vector potential A
        self.output_head    = o3.Linear(self.hidden_irreps, self.output_irreps)

        self.loss_fn = nn.MSELoss()

    @staticmethod
    def curl(field, coords):
        # field is the vector potential A (batch, seq, 3)
        # coords is (batch, seq, 3)
        Ax, Ay, Az = field[..., 0], field[..., 1], field[..., 2]
        X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]

        # Use central differences to approximate spatial derivatives along the sequence
        dAy_dZ = torch.gradient(Ay, spacing=(Z,), dim=1)[0]
        dAz_dY = torch.gradient(Az, spacing=(Y,), dim=1)[0]
        
        dAz_dX = torch.gradient(Az, spacing=(X,), dim=1)[0]
        dAx_dZ = torch.gradient(Ax, spacing=(Z,), dim=1)[0]

        dAx_dY = torch.gradient(Ax, spacing=(Y,), dim=1)[0]
        dAy_dX = torch.gradient(Ay, spacing=(X,), dim=1)[0]

        Bx = dAz_dY - dAy_dZ
        By = dAx_dZ - dAz_dX
        Bz = dAy_dX - dAx_dY

        B = torch.stack([Bx, By, Bz], dim=2)
        return B

    def forward(self, x_scalars, x_vectors, delta_t, coords):
        # ---------- CfC input prep ---------------------------------- ▲
        x_in  = torch.cat([x_scalars, x_vectors], dim=2)   # (B, T, F)
        delta = delta_t.squeeze(-1)                           # (B, T)
        h     = self.cfc(x_in, timespans=delta)               # (B, T, hidden_dim)
        # ------------------------------------------------------------

        h_eq = self.to_hidden(h)           # lift to equivariant space
        A    = self.output_head(h_eq)      # (B, T, 3)

        B    = self.curl(A, coords)        # div-free
        Bmag = torch.sqrt((B**2).sum(2, keepdim=True))
        return B, Bmag

    def training_step(self, batch, batch_idx):
        x_s, x_v, dt, y_s, y_v, coords = batch
        B_pred, B_mag_pred = self(x_s, x_v, dt, coords)
        loss = self.loss_fn(B_pred, y_v) + self.loss_fn(B_mag_pred, y_s)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_s, x_v, dt, y_s, y_v, coords = batch
        B_pred, B_mag_pred = self(x_s, x_v, dt, coords)
        vloss = self.loss_fn(B_pred, y_v) + self.loss_fn(B_mag_pred, y_s)
        self.log('val_loss', vloss, prog_bar=True)
        return vloss

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

        # fresh dataset for this fold
        ds    = CombinedDataset(flight_2, sequence_length=100)
        tr_ds = torch.utils.data.Subset(ds, tr_idx)
        va_ds = torch.utils.data.Subset(ds, va_idx)

        # μ/σ from training subset only
        def cat_seq(tensor, idx):
            return torch.cat([tensor[i:i+ds.sequence_length] for i in idx], 0)

        Xs_mu, Xs_sd = cat_seq(ds.X_scalars, tr_idx).mean(0), cat_seq(ds.X_scalars, tr_idx).std(0)
        Xv_mu, Xv_sd = cat_seq(ds.X_vectors, tr_idx).mean(0), cat_seq(ds.X_vectors, tr_idx).std(0)
        yv_mu, yv_sd = cat_seq(ds.y_vector , tr_idx).mean(0), cat_seq(ds.y_vector , tr_idx).std(0)
        ys_mu, ys_sd = cat_seq(ds.y_scalar , tr_idx).mean() , cat_seq(ds.y_scalar , tr_idx).std()
        co_mu, co_sd = cat_seq(ds.coords_t , tr_idx).mean(0), cat_seq(ds.coords_t , tr_idx).std(0)

        ds.X_scalars = (ds.X_scalars - Xs_mu) / (Xs_sd + 1e-8)
        ds.X_vectors = (ds.X_vectors - Xv_mu) / (Xv_sd + 1e-8)
        ds.y_vector  = (ds.y_vector  - yv_mu) / (yv_sd + 1e-8)
        ds.y_scalar  = (ds.y_scalar  - ys_mu) / (ys_sd + 1e-8)
        ds.coords_t  = (ds.coords_t  - co_mu) / (co_sd + 1e-8)

        # loaders
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=4)
        va_loader = DataLoader(va_ds, batch_size=32, num_workers=4)

        # model & trainer
        model = SE3_CfC_DivFree_Model(
            num_scalar_features=len(ds.scalar_features),
            num_vector_features=len(ds.vector_features) // 3
        )

        trainer = Trainer(max_epochs=50,
                          precision='16-mixed',
                          accelerator='gpu',
                          devices=1,
                          accumulate_grad_batches=2,
                          gradient_clip_val=1.0,
                          callbacks=[
                              RichProgressBar(),
                              ModelCheckpoint(dirpath=f'ckpt/fold{fold}',
                                              filename='best',
                                              monitor='val_loss',
                                              mode='min')
                          ])

        trainer.fit(model, tr_loader, va_loader)