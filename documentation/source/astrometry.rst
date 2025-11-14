Astrometry outputs and parameters
=================================

This page summarizes the astrometry options added to gulls, the new output columns, and the noise recipe used to generate observed astrometric positions.

Parameters
----------

- ``ASTROMETRY_ON`` (default: ``0``)
  - Enable astrometric computation and outputs when set to 1.
  - When 0, all sky-frame astrometry outputs are disabled and written as 0.0.

- ``ASTROMETRIC_SYS_FLOOR`` (units: mas, default: ``0.1``)
  - Per-axis systematic floor for astrometric uncertainty.
  - Combined in quadrature with the photon-limited term when producing per-epoch errors.

Coordinate systems and column names
-----------------------------------

All angles below are in milliarcseconds (mas) unless stated otherwise. Names match the lightcurve columns exactly.

Lens-frame (VBM) centroid
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``true_x_centroid`` (Einstein radii): Centroid x1 in the VBM lens frame (x1 along the binary axis).
- ``true_y_centroid`` (Einstein radii): Centroid x2 in the VBM lens frame (x2 perpendicular to the binary axis).

Sky NE (North/East) centroid — lens-centric offsets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``true_N_centroid_mas``: True (noise-free) North offset of the centroid in the sky frame.
- ``true_E_centroid_mas``: True (noise-free) East offset of the centroid in the sky frame.
- ``measured_N_centroid_mas``: Observed North centroid with noise applied, using flux-weighted blending of source and lens light.
- ``measured_E_centroid_mas``: Observed East centroid with noise applied, using flux-weighted blending of source and lens light.
- ``measured_N_centroid_error_mas``: 1-sigma per-axis uncertainty (North) for the observed centroid.
- ``measured_E_centroid_error_mas``: 1-sigma per-axis uncertainty (East) for the observed centroid.

Important: the NE offsets are lens-centric — i.e., they are centroid offsets relative to the lens origin, mapped to the local sky NE axes. They are not absolute coordinates.

Blending: The ``measured_*`` NE centroids represent the observed (flux-weighted) centroid of source+lens light. Because the lens is at the origin of the lens-centric frame, the lens contribution does not shift the lens-frame coordinates directly, but it down-weights the source-induced offset by a factor f_src = F_src / (F_src + F_lens) per epoch and band. The ``true_*`` NE centroids report the pure microlensing centroid of the source images (no lens-light blending).

Rotation from lens frame (x1/x2) to sky NE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let ``alpha`` be the trajectory-vs-axis angle in the event parameters, and let the on-sky position angle of the binary axis be ``phi_axis``. We define a rotation angle ``PosAng`` such that

``PosAng = phi_axis`` and the mapping from lens frame to NE is:

``N = x1 * cos(PosAng) - x2 * sin(PosAng)``

``E = x1 * sin(PosAng) + x2 * cos(PosAng)``

In practice, we determine ``phi_axis`` from either the parallax direction (when parallax is on) or the relative proper motion direction, and apply the offset with the event's ``alpha`` internally. If a reliable sky orientation cannot be established, NE outputs are zeroed (see Zeroing behavior).

Absolute RA/Dec centroid columns
--------------------------------

When astrometry is enabled, additional columns report the absolute sky position of the flux centroid in Equatorial (ICRS, J2000) coordinates per epoch. These combine the lens motion on the sky with the lens-centric NE centroid offsets:

- ``true_centroid_ra_deg``, ``true_centroid_dec_deg``: True (noise-free) absolute centroid coordinates in degrees.
- ``measured_centroid_ra_deg``, ``measured_centroid_dec_deg``: Observed centroid coordinates (noise applied) in degrees.
- ``measured_centroid_ra_error_deg``, ``measured_centroid_dec_error_deg``: 1-sigma uncertainties per axis in degrees.

Construction (small-angle mapping):

1. Base coordinates at t0: ``(RA0, Dec0)`` are the event coordinates.
2. Lens heliocentric proper motion drift (Equatorial components ``mu_RA*``, ``mu_Dec``):
   - ``Δt_years = (epoch - t0) / 365.25``
   - ``ΔE_PM = mu_RA* × Δt_years`` (mas), ``ΔN_PM = mu_Dec × Δt_years`` (mas)
3. Add annual parallax of the lens (observer-dependent): let ``(ΔN_par_AU, ΔE_par_AU)`` be the observer NE displacement in AU from the parallax module, and let the lens annual parallax be ``π_L`` in mas (for lens distance ``D_L`` in kpc, ``π_L = 1/D_L`` mas). Then

   ``ΔN_par = π_L × ΔN_par_AU`` (mas), ``ΔE_par = π_L × ΔE_par_AU`` (mas)

4. Total NE for absolute position (using the blended centroid for what an instrument measures):

   ``E_tot = E_blend_mas + ΔE_PM + ΔE_par`` and ``N_tot = N_blend_mas + ΔN_PM + ΔN_par``
4. Convert to degrees and add to base:
   - ``RA = RA0 + E_tot / (cos(Dec0) × 3600000)``
   - ``Dec = Dec0 + N_tot / 3600000``
5. The “measured” RA/Dec use the blended NE centroids and their per-axis errors, converted from mas to deg using the same small-angle relations. RA is wrapped into [0, 360).

Lens annual parallax: Included as described above by projecting the observer ephemeris into the event's NE basis and scaling by the lens annual parallax.

Notes on frames and units:
- ``true_x_centroid``/``true_y_centroid`` are in the lens frame (Einstein radii). NE columns are in the sky frame (mas) and lens-centric. Absolute RA/Dec are in degrees (ICRS/J2000) and represent the position on the sky of the flux centroid.

Noise model and zeroing rules
-----------------------------

Per-epoch astrometric uncertainties and observed values are generated as follows:

1. Compute a photometric signal-to-noise ratio per epoch from the simulated photometry:
   - ``SNR = |Aobs| / max(Aerr, 1e-12)``
2. PSF width is taken from the instrument model as ``FWHM`` in arcsec; we convert to mas via ``FWHM_mas = 1000 * FWHM``.
3. Photon-limited per-axis precision (1D) is approximated as:
   - ``sigma_photon = FWHM_mas / SNR``
4. Total per-axis uncertainty combines the photon term with a systematic floor:
   - ``sigma_axis = sqrt(sigma_photon^2 + ASTROMETRIC_SYS_FLOOR^2)``
5. Observed NE centroids are produced by adding independent Gaussian noise to the true NE centroids:
   - ``obs_N = true_N + Normal(0, sigma_axis)``
   - ``obs_E = true_E + Normal(0, sigma_axis)``
   - ``obs_N_centroid_err_mas = obs_E_centroid_err_mas = sigma_axis``

Zeroing behavior (no sky orientation or disabled):
- If ``ASTROMETRY_ON = 0``, NE and absolute RA/Dec astrometric outputs are disabled (set to 0.0).
- If no reliable sky orientation is available (e.g., neither parallax direction nor relative proper motion direction), the NE outputs are set to 0.0 for that event and absolute RA/Dec are not constructed.
- In ideal photometry mode, the measured NE centroids equal the true NE centroids and their errors are set to the systematic floor (no random noise added).

Summary of absolute construction
--------------------------------

- NE outputs (true/measured) are lens-centric centroid offsets rotated into NE using the mapping above; measured NE is flux-blended (source+lens).
- Absolute RA/Dec = base (RA0, Dec0) + lens PM drift (NE) + lens annual parallax (NE) + blended NE centroid offset, mapped to RA/Dec via the small-angle relations.
