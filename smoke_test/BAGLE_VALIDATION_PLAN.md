# BAGLE Validation Plan for Gulls Astrometry

## Overview

This document outlines the strategy for validating the astrometry implementation in gulls_mp against BAGLE_Microlensing. The goal is to ensure that the physics of the astrometric centroid shift calculations are correct, particularly:

1. **Coordinate transformations**: From lens frame (x1, x2) to sky frame (North, East)
2. **Centroid shift physics**: The microlensing-induced shift in the image centroid
3. **Reference frames**: Understanding what coordinate system we're working in

## What We're Validating

### ✅ What Should Match (approximately)
- **Magnification curves**: Should be nearly identical (RMS < 1e-6)
- **Centroid shift in N/E frame**: Should be similar, accounting for:
  - Different parallax implementations (L2 approximation vs JPL Horizons)
  - Potential sub-mas differences due to numerical methods
  - Expected RMS differences: < 0.1 mas for most cases

### ❓ What We're NOT Yet Validating
- **Absolute astrometry** (RA, Dec): This requires:
  - Source baseline position (RA, Dec at reference epoch)
  - Proper motion vectors
  - Conversion from relative centroid shift to absolute position
  - This is a separate validation step (coming next)

## Current Implementation Status

### In `pllxLightcurveGenerator.cpp`

The current code does:

1. **Computes lens-frame centroid** (Einstein radii):
   ```cpp
   double cx = Event->vbm->astrox1;  // x1 (along binary axis)
   double cy = Event->vbm->astrox2;  // x2 (perpendicular to binary axis)
   ```

2. **Rotates to sky frame (N/E)**:
   ```cpp
   double cx_mas = cx * Event->thE;  // Convert to mas
   double cy_mas = cy * Event->thE;
   double cN = cx_mas * cosPos + cy_mas * sinPos;   // North
   double cE = -cx_mas * sinPos + cy_mas * cosPos;  // East
   ```

3. **PosAng determination**:
   - Uses parallax vector direction if available: `phi_pi = atan2(piEE, piEN)`
   - Falls back to relative proper motion direction if no parallax
   - Applies: `PosAng = phi - alpha + dPosAng`

### What's Stored

- `Event->cNtrue[idx]`, `Event->cEtrue[idx]`: **Centroid shift in mas** (N/E frame)
- These are **relative shifts**, not absolute positions
- They represent the offset of the lensed image centroid from the unlensed source position

## Validation Script: `validation_bagle.py`

### What It Does

1. **Reads gulls lightcurve file** with header parameters
2. **Extracts physical parameters** and converts to BAGLE format
3. **Creates matching BAGLE model** (PSPL with/without parallax)
4. **Computes both predictions**:
   - Magnification: `A(t)`
   - Centroid shift: `[δN, δE](t)` in mas
5. **Compares and reports**:
   - RMS differences
   - Max differences
   - Diagnostic plots

### Required Parameters in Gulls Header

The validation script needs these parameters in the `.lc` file header:

```cpp
# lens_mass = 10.0         # Msun
# t0 = 57000.0             # MJD
# u0 = 0.5                 # dimensionless (impact parameter in thetaE)
# theta_e = 1.0            # mas (Einstein radius)
# lens_dist = 4000.0       # pc
# source_dist = 8000.0     # pc
# muL_E = 0.0              # mas/yr (lens proper motion East)
# muL_N = -7.0             # mas/yr (lens proper motion North)
# muS_E = 5.0              # mas/yr (source proper motion East)
# muS_N = 0.0              # mas/yr (source proper motion North)
# xS0_E = 0.0              # arcsec (source position at t0, East)
# xS0_N = 0.0              # arcsec (source position at t0, North)
# b_sff = 1.0              # source flux fraction
# mag_src = 19.0           # source magnitude
# ra = 269.94              # deg (for parallax models)
# dec = -28.64             # deg (for parallax models)
# alpha = 45.0             # deg (trajectory angle)
```

### Usage

```bash
# Basic usage (no parallax)
python smoke_test/validation_bagle.py smoke_test/output/std/smoke_std_0_1.det.lc

# With parallax
python smoke_test/validation_bagle.py smoke_test/output/std/smoke_std_0_1.det.lc --parallax

# Specify output directory
python smoke_test/validation_bagle.py smoke_test/output/std/smoke_std_0_1.det.lc \
    --output-dir smoke_test/validation_results/
```

## Implementation Steps

### Step 1: Add Parameters to Gulls Output ⚠️ TODO

**File**: `src/outputLightcurve.cpp`

Add these lines in the header output section (after line ~100):

```cpp
// Add validation parameters for BAGLE comparison
fprintf(lcfile_ptr, "# lens_mass = %.6f\n", Lenses->data[Event->lens][Lenses->MASS]);
fprintf(lcfile_ptr, "# lens_dist = %.6f\n", Event->dL);
fprintf(lcfile_ptr, "# source_dist = %.6f\n", Event->dS);
fprintf(lcfile_ptr, "# t0 = %.12f\n", Event->t0);
fprintf(lcfile_ptr, "# u0 = %.6f\n", Event->u0);
fprintf(lcfile_ptr, "# theta_e = %.6f\n", Event->thE);
fprintf(lcfile_ptr, "# tE = %.6f\n", Event->tE_r);
fprintf(lcfile_ptr, "# alpha = %.6f\n", Event->alpha);

// Proper motions (if available)
fprintf(lcfile_ptr, "# muL_E = %.6f\n", Event->muL_E);  // Need to add these to Event struct
fprintf(lcfile_ptr, "# muL_N = %.6f\n", Event->muL_N);
fprintf(lcfile_ptr, "# muS_E = %.6f\n", Event->muS_E);
fprintf(lcfile_ptr, "# muS_N = %.6f\n", Event->muS_N);

// Parallax parameters (if PARALLAX=1)
if (Paramfile->pllxMultiplyer) {
    fprintf(lcfile_ptr, "# piEN = %.6f\n", Event->piEN);
    fprintf(lcfile_ptr, "# piEE = %.6f\n", Event->piEE);
    fprintf(lcfile_ptr, "# ra = %.8f\n", Event->ra * TO_DEG);
    fprintf(lcfile_ptr, "# dec = %.8f\n", Event->dec * TO_DEG);
}

// Source position (if available)
fprintf(lcfile_ptr, "# xS0_E = %.12f\n", 0.0);  // Placeholder
fprintf(lcfile_ptr, "# xS0_N = %.12f\n", 0.0);  // Placeholder

// Photometric parameters
fprintf(lcfile_ptr, "# b_sff = %.6f\n", Event->fs[0]);  // Source flux fraction
fprintf(lcfile_ptr, "# mag_src = %.6f\n", Sources->mags[Event->source][World[0].filter]);
```

**Note**: Some of these parameters (like individual muL, muS components) may need to be stored in the Event structure if they're not already available. Currently, gulls might only store the relative proper motion.

### Step 2: Run Smoke Tests

```bash
cd /Users/malpas.1/Code/gulls_mp
python smoke_test/run_smoke_test.py --cases smoke_std
```

This will generate `.lc` files in `smoke_test/output/std/` with the required headers.

### Step 3: Run BAGLE Validation

```bash
# Make sure BAGLE is in your Python path
export PYTHONPATH="/Users/malpas.1/Code/BAGLE_Microlensing/src:$PYTHONPATH"

# Run validation on generated lightcurves
python smoke_test/validation_bagle.py smoke_test/output/std/smoke_std_0_*.det.lc
```

### Step 4: Interpret Results

#### Good Results (Physics is Correct)
- **Magnification**: RMS < 1e-6, relative RMS < 1e-8
- **Centroid shift**: RMS < 0.1 mas for typical events
  - Larger differences (0.1-1 mas) acceptable if:
    - Event has strong parallax (L2 vs JPL difference)
    - Event near edges of observing season
  
#### Bad Results (Something is Wrong)
- **Large magnification differences** (RMS > 1e-4): 
  - Check VBM settings (tolerance, accuracy)
  - Check parameter conversions
- **Large centroid differences** (RMS > 1 mas):
  - Rotation matrix is wrong (check PosAng calculation)
  - Units are wrong (Einstein radii vs mas vs arcsec)
  - Reference frame mismatch (heliocentric vs geocentric)

### Step 5: Fix Issues

Based on the validation results:

1. **If rotation is wrong**:
   - Check `PosAng` calculation in `pllxLightcurveGenerator.cpp`
   - Verify `phi_pi = atan2(piEE, piEN)` convention
   - Check rotation matrix: `[N, E] = R(PosAng) * [x1, x2]`

2. **If reference frame is wrong**:
   - Check if we're using heliocentric vs geocentric coordinates consistently
   - Verify parallax vector direction convention

3. **If units are wrong**:
   - Verify conversion from Einstein radii to mas: `cx_mas = cx * thE`
   - Check that `thE` is in mas (not arcsec)

## Expected Differences: Gulls vs BAGLE

### Sources of Acceptable Differences

1. **Parallax Implementation**:
   - Gulls: L2 orbit approximation
   - BAGLE: JPL Horizons ephemeris
   - Expected difference: < 0.1 mas typically, up to 1 mas for extreme geometries

2. **Numerical Methods**:
   - VBMicrolensing vs BAGLE's internal methods
   - Different tolerances and algorithms
   - Expected difference: < 0.01 mas for magnification > 1.1

3. **Binary Systems**:
   - Gulls: Full binary lens calculation
   - This validation: PSPL only
   - **Action**: Create separate BSPL/PSBL validation later

### Sources of Unacceptable Differences

1. **Coordinate frame errors**: > 10% of signal
2. **Sign errors**: Complete inversion of N or E component
3. **Unit errors**: Factor of 1000 (mas vs arcsec) or other large factors
4. **Rotation errors**: Components mixed or rotated by wrong angle

## Next Steps: Absolute Astrometry Validation

Once centroid shift validation passes, we need to validate absolute astrometry:

1. **Define baseline source position**: (RA₀, Dec₀) at reference epoch
2. **Add proper motion**: (RA₀ + μ_RA * Δt, Dec₀ + μ_Dec * Δt)
3. **Add centroid shift**: (RA, Dec) = baseline + PM + shift
4. **Convert to degrees** with correct cos(Dec) factor for RA
5. **Validate against BAGLE's `get_astrometry()`** method

This is currently done in `outputLightcurve.cpp` lines 375-410, but needs validation.

## References

### BAGLE Documentation
- BAGLE Tutorial: `/Users/malpas.1/Code/BAGLE_Microlensing/BAGLE_TUTORIAL.txt`
- Model Fitting: `/Users/malpas.1/Code/BAGLE_Microlensing/model_fit_tutorial.txt`
- Frame Conventions: `/Users/malpas.1/Code/BAGLE_Microlensing/Frame_conversion_examples.txt`

### Gulls Implementation
- Lightcurve generation: `src/pllxLightcurveGenerator.cpp`
- Output formatting: `src/outputLightcurve.cpp`
- Smoke tests: `smoke_test/run_smoke_test.py`

### VBMicrolensing
- Repository: `/Users/malpas.1/Code/VBMicrolensing`
- Astrometry: Uses `astrox1`, `astrox2` for centroid in lens frame

## Questions to Answer

- [ ] Do we have proper motion components (muL, muS) stored separately?
- [ ] What is the reference frame for source/lens positions? (Galactic? Equatorial?)
- [ ] Are we using heliocentric or geocentric parallax consistently?
- [ ] Do we need to add any coordinate transformation utilities?
- [ ] Should we create a BSPL/PSBL validation separately?

## Success Criteria

✅ **Phase 1 Complete** when:
- Magnification RMS < 1e-6 for all test cases
- Centroid shift RMS < 0.1 mas for no-parallax cases
- Centroid shift RMS < 1.0 mas for parallax cases
- Visual inspection of residual plots shows no systematic offsets

✅ **Phase 2 Complete** when:
- Absolute astrometry (RA, Dec) validated against BAGLE
- Proper motion vectors correctly applied
- cos(Dec) factors correctly applied to RA
- Reference frame transformations validated
