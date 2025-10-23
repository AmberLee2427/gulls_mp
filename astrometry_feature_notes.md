# Astrometry Feature Additions in `gulls_mp`

This document records the astrometric light-curve machinery that was introduced after branching from `dev`.  The focus is on the mathematical model, reference-frame bookkeeping, and observable products, with explicit call-outs for places where the current implementation looks fragile or incorrect.

## 1. Scope and Control Surface

- New runtime switches are parsed from the parameter file (`ASTROMETRY_ON`, `ASTROMETRIC_SYS_FLOOR`; `src/readParamfile.cpp:234`) and cached on the simulation state (`struct filekeywords`, `src/structures.h:140`).
- Event buffers now carry sky-frame centroid tracks and uncertainties (`cNtrue`, `cEtrue`, etc.; `src/structures.h:386`–`392`), which are resized alongside the legacy photometry arrays (`src/timeSequencer.cpp:231`–`257`).
- The main generator toggles VBMicrolensing’s internal centroid solver through `Event->vbm->astrometry` (`src/pllxLightcurveGenerator.cpp:57`) and provides the data needed to rotate lens-frame centroids into the observable North/East frame.
- Downstream outputs and plotting utilities consume these new columns:
  - `outputLightcurve.cpp` appends NE centroids and absolute RA/Dec tracks to the `.lc` products (`src/outputLightcurve.cpp:380`–`436`).
  - `smoke_test/plotting.py` overlays VBMicrolensing predictions and draws diagnostic vectors for the proper-motion budget (`smoke_test/plotting.py:208`–`269`, `smoke_test/plotting.py:536`–`638`).

## 2. Lens-Frame Modelling

Let the intrinsic event parameters be the usual binary-lens tuple \((s, q, u_0, \alpha, \rho, t_E, t_0)\) augmented by the Einstein radius \(\theta_E\) and the source/relative proper-motion metadata already present in GULLS.

### 2.1 Source trajectory in the lens plane

For an epoch \(t_i\) the scaled time coordinate is
\[
  \tau_i = \frac{t_i - t_0}{t_E^{(\mathrm{ref})}} + \delta\tau_i,
\]
where the parallax correction
\(
  \delta\tau_i = \Delta t_{\mathrm{par},i}
\)
is injected when `PARALLAX` is enabled (`src/pllxLightcurveGenerator.cpp:116`–`123`).  The impact-parameter component is
\[
  u_i = u_0 + \delta u_{\mathrm{par},i}.
\]

Coordinates in the binary lens frame (with \(\hat{x}_1\) along the lens axis and \(\hat{x}_2\) perpendicular) follow the standard rotation by the trajectory angle \(\alpha\):
\[
\begin{aligned}
  x_{s,i} &= \tau_i \cos\alpha - u_i \sin\alpha + x_{\mathrm{CoM}},\\
  y_{s,i} &= \tau_i \sin\alpha + u_i \cos\alpha,
\end{aligned}
\]
where the barycentric shift of the primary lens is \(x_{\mathrm{CoM}} = -(1-m_1) s\) (`src/pllxLightcurveGenerator.cpp:117`–`134`).

### 2.2 VBMicrolensing interface

With `Event->vbm->astrometry` flagged, `BinaryMag2` returns both magnification and the instantaneous centroid in Einstein radii:
\[
  (\xi_1, \xi_2) = (\texttt{Event->vbm->astrox1},\ \texttt{Event->vbm->astrox2}).
\]

For single sources the lens-frame centroid is simply \(\boldsymbol{\xi}\) (`src/pllxLightcurveGenerator.cpp:200`–`209`).  For a binary source the code re-evaluates VBMicrolensing for the companion trajectory and forms the flux-weighted centroid
\[
\boldsymbol{\xi}_{\mathrm{tot}} = \frac{F_1\,\boldsymbol{\xi}_1 + F_2\,\boldsymbol{\xi}_2}{F_1 + F_2},
\]
with \(F_1 = A_1\) and \(F_2 = f_{s,2/1} A_2\) (`src/pllxLightcurveGenerator.cpp:162`–`176`).  In all cases the lens-frame centroid is stored (for smoke tests) as `xctrue`, `yctrue`.

## 3. Rotation into the North/East frame

The Einstein-radius scaling converts to milliarcseconds:
\[
  (c_1, c_2) = \theta_E\,(\xi_1, \xi_2).
\]

To project into geocentric North/East coordinates the code solves for a rotation angle \(\Phi\) between the lens axis and North:

1. Prefer the direction of \(\boldsymbol{\pi}_E\):
   \[
     \Phi = \operatorname{atan2}(\pi_{E,E}, \pi_{E,N}) - \alpha.
   \]
   This path is taken when `PARALLAX` is active (`src/pllxLightcurveGenerator.cpp:62`–`69`).
2. Otherwise fallback to the relative proper-motion vector.  The Galactic components \((\mu_{\ell}, \mu_b)\) are converted to Equatorial \((\mu_{\alpha^*}, \mu_\delta)\) via `coords.mulb2ad` (`src/pllxLightcurveGenerator.cpp:71`–`78`).  \(\Phi\) is then set to \(\operatorname{atan2}(\mu_{\alpha^*}, \mu_\delta) - \alpha\).

If neither vector is available the rotation is left at the identity and astrometric products are suppressed (`astrom_ok = false`).

Using \(\Phi\), the code applies the right-handed rotation matrix
\[
  R(\Phi) =
  \begin{pmatrix}
    \cos\Phi & \sin\Phi \\
   -\sin\Phi & \cos\Phi
  \end{pmatrix},
\]
so that
\[
  \begin{pmatrix}\Delta N \\ \Delta E\end{pmatrix}
  = R(\Phi)\,
    \begin{pmatrix}c_1 \\ c_2\end{pmatrix}.
\]
The results are cached in `cNtrue`, `cEtrue` (`src/pllxLightcurveGenerator.cpp:221`–`226`), while `cNobs`, `cEobs` start as noiseless copies.

## 4. Absolute sky coordinates and light-curve output

Prior to writing the `.lc` file the code adds the lens heliocentric proper motion to the microlensing offsets and converts into RA/Dec:
\[
\begin{aligned}
  \Delta N_{\text{tot}}(t_i) &= \Delta N_{\text{micro}}(t_i) + \mu_\delta\ \Delta t_i,\\
  \Delta E_{\text{tot}}(t_i) &= \Delta E_{\text{micro}}(t_i) + \mu_{\alpha^*}\ \Delta t_i,
\end{aligned}
\]
with \(\Delta t_i = (t_i - t_0)/365.25\ \text{yr}\) (`src/outputLightcurve.cpp:380`–`386`).

Assuming small angles,
\[
\begin{aligned}
  \alpha_{\text{true}}(t_i) &= \alpha_0 + \frac{\Delta E_{\text{tot}}(t_i)}{\cos\delta_0}\,\frac{1}{3600\cdot10^3},\\
  \delta_{\text{true}}(t_i) &= \delta_0 + \frac{\Delta N_{\text{tot}}(t_i)}{3600\cdot10^3},
\end{aligned}
\]
where \(\alpha_0,\delta_0\) are the event coordinates from the Galactic-to-equatorial transform (`src/timeSequencer.cpp:20`–`33`).  Observed quantities reuse the same transformation with `cEobs`, `cNobs`, and the positional errors reuse the same scaling (`src/outputLightcurve.cpp:394`–`401`).  All of these arrays are streamed to the light-curve file with explicit column names (`src/outputLightcurve.cpp:409`–`436`).

## 5. Noise and systematics model

After photometry, the astrometric measurement is perturbed by an axisymmetric Gaussian whose width combines photon statistics with a configurable floor:
\[
  \sigma_{\text{axis}} = \sqrt{\left[\frac{\mathrm{FWHM}}{\sqrt{8\ln 2}}\,\sigma_{\text{phot}}\right]^2 + \sigma_{\text{sys}}^2},
\]
where \(\sigma_{\text{phot}} = | \sigma_F / F | \) is derived from the measured flux errors and \(\sigma_{\text{sys}}\) is `ASTROMETRIC_SYS_FLOOR` (`src/photometry.cpp:134`–`152`).  The observed centroids are then drawn as
\[
  c_{N,\text{obs}} = c_{N,\text{true}} + \sigma_{\text{axis}}\,\mathcal{N}(0,1),
\quad
  c_{E,\text{obs}} = c_{E,\text{true}} + \sigma_{\text{axis}}\,\mathcal{N}(0,1),
\]
with matching 1-σ errors reported (`src/photometry.cpp:154`–`158`).  In “ideal” photometry mode the code bypasses the random deviate and simply forwards the truth values while pinning the error budget to the systematic floor (`src/photometry.cpp:102`–`121`).

> **Open bug:** the noise draw is skipped whenever \(c_{N,\text{true}} = c_{E,\text{true}} = 0\).  Exact symmetry points (e.g. at large \(|t|\) or during axis crossings) therefore record zero observed shift with zero uncertainty, even though centroiding noise should still be present (`src/photometry.cpp:154`–`164`).  This yields discontinuities in the time series and biases the RA/Dec errors.

## 6. Smoke-test instrumentation

The smoke test ingests the new columns to produce three coordinated diagnostics per light curve:

1. **Lens-frame overlay.**  `BinaryAstroLightCurve` from VBMicrolensing is evaluated a second time with the summary metadata and compared against the stored lens-frame truth (`smoke_test/plotting.py:208`–`269`).  Because the frame mapping is uncertain, the script brute-forces all eight permutations/sign flips of the VBM output in search of the smallest MSE difference from the simulated centroid (`smoke_test/plotting.py:233`–`256`).  This hack is a strong indicator that the rotation logic above is still inconsistent with the VBMicrolensing convention.

2. **North/East panel.**  `true_N/E` and `measured_N/E` are plotted alongside the VBM sky-track, using the new error bars (`smoke_test/plotting.py:559`–`614`).

3. **Absolute RA/Dec panel.**  The code reconstructs the absolute sky track and draws diagnostic vectors for the geocentric relative motion, plus the heliocentric source and lens proper motions when available (`smoke_test/plotting.py:536`–`556`, `smoke_test/plotting.py:620`–`638`).  Direction arrows span the temporal extent of the light curve to make frame mix-ups easier to spot.

Any failure to build the VBM overlay when astrometry is expected triggers the smoke test (`smoke_test/plotting.py:870`–`909`).

## 7. Additional gaps and risks

1. **Frame alignment remains ambiguous.**  
   The need for the sign/permutation search in the smoke test (`smoke_test/plotting.py:233`–`256`) implies that the rotation angle \(\Phi\) may still be defined in the wrong sense or with a missing reflection.  Comparing against the BAGLE frame-conversion notebooks—particularly `BAGLE_Microlensing/docs/frame_convert.rst`—would help pin down whether \(\hat{x}_1\) should map to North or East after subtracting \(\alpha\).

2. **Zeroing out astrometry when orientation is unknown.**  
   If neither \(\boldsymbol{\pi}_E\) nor \(\boldsymbol{\mu}_{\rm rel}\) is available, `astrom_ok` stays false and all centroids are forced to zero (e.g. events without parallax or catalog proper motions; `src/pllxLightcurveGenerator.cpp:80`–`89`).  The `.lc` file still carries the lens proper-motion drift, so the RA/Dec columns evolve while the NE offsets are identically zero: this mixture is confusing for downstream inference.

3. **Multiple-source weighting ignores baseline mixing.**  
   The flux-weighted centroid uses \(F_1=A_1\) and \(F_2=f_{s,2/1} A_2\) (`src/pllxLightcurveGenerator.cpp:168`–`172`).  The total flux in the denominator is \(A_1 + f_{s,2/1} A_2\), whereas the photometric magnification stored in `Atrue` subtracts the baseline contribution for source 2 (`amp + f_s (A_2-1)`).  This mismatch can bias the centroid when the secondary contributes significant baseline light.

4. **Proper-motion conversion assumptions.**  
   The conversion from Galactic to Equatorial proper motion relies on `coords.mulb2ad` returning \(\mu_{\alpha^*}\) (i.e. already multiplied by \(\cos\delta\); `src/classes/coords.cpp:272`–`294`).  If the upstream catalog stores \(\mu_\alpha\) without the cosine factor the RA drift will be off by \(\cos\delta\).  Cross-checking against the Astrometry repo’s derivations (`Astrometry/docs/astrometry.tex`) is recommended.

5. **RA/Dec suppression when astrometry off.**  
   When `ASTROMETRY_ON=0`, the `.lc` columns for RA/Dec default to zero (`src/outputLightcurve.cpp:388`–`407`).  Downstream tooling that expects absolute coordinates will therefore see the event at the origin unless those columns are explicitly ignored.

## 8. Suggested validation steps

1. Use the BAGLE `Frame_conversion_examples.ipynb` to derive the expected NE centroids for a known configuration and compare against GULLS output with identical parameters.
2. Modify the smoke test to log the best permutation it selects; this will expose the systematic sign needed to fix the rotation.
3. Inject a synthetic event where \(\Delta N = \Delta E = 0\) for all times but request astrometry to verify that the noise model produces finite uncertainties; this will confirm the bug in Section 5 before patching.
4. Cross-check the RA/Dec drift against the Astrometry repo’s analytic model (equations in `Astrometry/docs/astrometry.tex`) to ensure the lens proper motion is not double-counted.

---

**References**

- VBMicrolensing centroid solver: `src/classes/VBMicrolensingLibrary.cpp` (numerous `astrox{1,2}` code paths).
- Error-model derivation followed from Astrometry collaboration notes (`Astrometry/docs/astrometry.tex`).
- Frame-convention comparisons are available in `BAGLE_Microlensing/docs/frame_convert.rst`.

---

**Figures**

![smoke_test/output/croin/smoke_croin_binary/smoke_croin_binary_0_0_0.det_lensframe_plot.png](smoke_test/output/croin/smoke_croin_binary/smoke_croin_binary_0_0_0.det_lensframe_plot.png)

![smoke_test/output/croin/smoke_croin_binary/smoke_croin_binary_0_0_0.det_plot.png](smoke_test/output/croin/smoke_croin_binary/smoke_croin_binary_0_0_0.det_plot.png)