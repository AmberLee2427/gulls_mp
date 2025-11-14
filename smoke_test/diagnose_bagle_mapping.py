import numpy as np, pandas as pd, math
from pathlib import Path

base = Path(__file__).resolve().parents[0]
lc = base / 'output' / 'std' / 'smoke_std' / 'smoke_std_0_0_0.all.lc'
side = base / 'output' / 'std' / 'smoke_std' / 'smoke_std_0_0_0.all_bagle.npz'
print('LC:', lc.resolve())
print('Sidecar:', side.resolve())

df = pd.read_csv(lc, sep='\s+', comment='#', header=0)
z = np.load(side)

# extract
N = df['true_N_centroid_mas'].to_numpy(dtype=float)
E = df['true_E_centroid_mas'].to_numpy(dtype=float)
true_x = df['true_x_centroid'].to_numpy(dtype=float)
true_y = df['true_y_centroid'].to_numpy(dtype=float)
src_x = df['source_x'].to_numpy(dtype=float)
src_y = df['source_y'].to_numpy(dtype=float)

# rel true (ER)
try:
    thetaE = float(np.asarray(z['thetaE_mas']).reshape(-1)[0])
except Exception:
    thetaE = float(z['thetaE_mas'])
rel_true_x = (true_x - src_x)
rel_true_y = (true_y - src_y)

mask = np.isfinite(N) & np.isfinite(E) & np.isfinite(rel_true_x) & np.isfinite(rel_true_y)
N = N[mask]; E = E[mask]; rel_true_x = rel_true_x[mask]; rel_true_y = rel_true_y[mask]
print('samples used:', N.size)

# convert to ER
n_er = N / thetaE
e_er = E / thetaE
X = np.vstack([n_er, e_er]).T
Y = np.vstack([rel_true_x, rel_true_y]).T

# fit linear transform: Y = X @ M^T  => np.linalg.lstsq(X, Y) returns M
M, *_ = np.linalg.lstsq(X, Y, rcond=None)
print('\nLeast-squares M (maps [n,e] -> [x,y]):')
print(M)

# examine whether M is close to scaled rotation
u, svals, vh = np.linalg.svd(M)
print('\nSVD singular values:', svals)
print('det(M)=', np.linalg.det(M))

# extract c,s from M first row
c = float(M[0,0]); s = float(M[0,1])
phi_est_deg = (math.degrees(math.atan2(s, c))) % 360
print('phi_est (deg)=', phi_est_deg)

# orthonormality check on M normalized columns
col0 = M[:,0]; col1 = M[:,1]
norm0 = np.linalg.norm(col0); norm1 = np.linalg.norm(col1)
print('col norms:', norm0, norm1)
print('col dot product:', np.dot(col0, col1))

# load bagle shifts
bagle_E = np.asarray(z['bagle_shift_E_mas'], dtype=float)
bagle_N = np.asarray(z['bagle_shift_N_mas'], dtype=float)

# compute overlay using current plotting convention (no flips)
n_er_b = bagle_N / thetaE
e_er_b = bagle_E / thetaE
x_b = c * n_er_b + s * e_er_b
y_b = -s * n_er_b + c * e_er_b

# align mask for bagle arrays
mask2 = np.isfinite(x_b) & np.isfinite(y_b) & np.isfinite(rel_true_x) & np.isfinite(rel_true_y)
print('\nOverlay comparison samples:', mask2.sum())
rmse_default = math.sqrt(np.mean((x_b[mask2]-rel_true_x[mask2])**2 + (y_b[mask2]-rel_true_y[mask2])**2))
print('RMSE default (bagle as E,N; plotting R):', rmse_default)

# diagnostics: try flipping bagle_N sign (diagnostic only)
x_b2 = c * (-n_er_b) + s * e_er_b
y_b2 = -s * (-n_er_b) + c * e_er_b
rmse_flipN = math.sqrt(np.mean((x_b2[mask2]-rel_true_x[mask2])**2 + (y_b2[mask2]-rel_true_y[mask2])**2))
print('RMSE flip N sign (diagnostic):', rmse_flipN)

# try swapping components (treat bagle as N,E)
x_b3 = c * e_er_b + s * n_er_b
y_b3 = -s * e_er_b + c * n_er_b
rmse_swap = math.sqrt(np.mean((x_b3[mask2]-rel_true_x[mask2])**2 + (y_b3[mask2]-rel_true_y[mask2])**2))
print('RMSE swap components (diagnostic):', rmse_swap)

# try swap+flip
x_b4 = c * (-e_er_b) + s * (-n_er_b)
y_b4 = -s * (-e_er_b) + c * (-n_er_b)
rmse_swapflip = math.sqrt(np.mean((x_b4[mask2]-rel_true_x[mask2])**2 + (y_b4[mask2]-rel_true_y[mask2])**2))
print('RMSE swap+flip (diagnostic):', rmse_swapflip)

# small summary
print('\nSummary:')
print(' - M appears to have scale factors, not a pure rotation (SVD sing vals above).')
print(' - phi_est (deg)=', phi_est_deg)
print(' - RMSEs (ER): default, flipN, swap, swap+flip =', rmse_default, rmse_flipN, rmse_swap, rmse_swapflip)
print('\nNote: these diagnostic permutations are computed only to help find the origin of the sign/order mismatch; they do NOT alter sidecars or any repo files.')
