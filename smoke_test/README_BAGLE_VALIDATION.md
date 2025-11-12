# BAGLE Validation for Gulls Astrometry

This directory contains tools for validating the astrometry implementation in gulls_mp against BAGLE_Microlensing.

## Quick Start

### Option 1: Simple Reference Test (Recommended First)

Run this to generate BAGLE reference predictions without needing gulls output:

```bash
cd /Users/malpas.1/Code/gulls_mp/smoke_test
python validation_bagle_simple.py
```

This will:
- Create a PSPL model with known parameters
- Compute magnification and astrometric predictions
- Generate reference plots showing what gulls should produce
- Export reference data for comparison

**Output**: `validation_output/bagle_reference_*.png` and `.txt`

### Option 2: Full Validation Against Gulls Output

Once you have gulls lightcurve files with proper headers:

```bash
python validation_bagle.py /path/to/gulls_lightcurve.lc
```

This will:
- Parse gulls lightcurve parameters
- Create matching BAGLE model
- Compare predictions
- Generate diagnostic plots showing differences

## Files

### Validation Scripts

- **`validation_bagle_simple.py`**: 
  - Standalone reference generator
  - Doesn't require gulls output
  - Good for understanding expected behavior
  - Run this first!

- **`validation_bagle.py`**:
  - Full comparison script
  - Requires gulls `.lc` files with headers
  - Generates difference plots and statistics
  - Use after modifying gulls output

### Documentation

- **`BAGLE_VALIDATION_PLAN.md`**:
  - Comprehensive validation strategy
  - What to validate and why
  - Expected differences
  - Implementation steps
  - Troubleshooting guide

## Current Status

### ✅ What's Implemented in Gulls

1. **Centroid calculation** in lens frame (VBMicrolensing)
2. **Rotation to sky frame** (N/E coordinates)
3. **PosAng calculation** from parallax or proper motion vectors
4. **Output columns** for astrometry (true_N_centroid_mas, true_E_centroid_mas)

### ⚠️ What Needs Validation

1. **Coordinate transformation** from lens frame to sky frame
2. **Sign conventions** (N/E, trajectory angle)
3. **Units** (Einstein radii → mas → arcsec)
4. **Reference frame** consistency (heliocentric vs geocentric)

### 🔧 What Needs Implementation

1. **Header parameters** in gulls `.lc` output:
   - Physical parameters (mL, dL, dS, etc.)
   - Proper motion components (muL_E, muL_N, muS_E, muS_N)
   - Derived quantities (theta_e, tE)
   - See `BAGLE_VALIDATION_PLAN.md` for full list

2. **Proper motion storage** in Event structure (may already exist)

3. **Absolute astrometry validation** (Phase 2, after centroid validation)

## Physics Check: What We're Validating

### The Coordinate Transformations

```
VBMicrolensing (lens frame)
    astrox1, astrox2  [Einstein radii]
           ↓
    × thetaE  [convert to mas]
           ↓
Rotation by PosAng
    [x1, x2] → [N, E]  [mas, sky frame]
           ↓
Add proper motion
    + μ * Δt  [mas]
           ↓
Convert to RA/Dec
    → (RA, Dec)  [degrees]
```

### Key Questions

1. **Is PosAng calculated correctly?**
   - From parallax: `phi_pi = atan2(piEE, piEN)`
   - From proper motion: `phi_mu = atan2(muRel_E, muRel_N)`
   - Combined: `PosAng = phi - alpha + dPosAng`

2. **Is the rotation matrix correct?**
   ```cpp
   cN = cx_mas * cos(PosAng) + cy_mas * sin(PosAng)
   cE = -cx_mas * sin(PosAng) + cy_mas * cos(PosAng)
   ```

3. **Are we in the right reference frame?**
   - Source position: heliocentric or geocentric?
   - Parallax vector: which direction?
   - Proper motions: relative or absolute?

## Expected Results

### Good Validation ✅

- **Magnification**: RMS difference < 1e-6
- **Centroid shift** (no parallax): RMS < 0.1 mas
- **Centroid shift** (with parallax): RMS < 1.0 mas
  - Larger acceptable due to L2 vs JPL differences
- **Visual inspection**: No systematic offsets in N or E

### Common Issues ❌

1. **Sign flip**: Check if N or E component is inverted
   - Wrong sign in rotation matrix
   - Wrong parallax vector convention

2. **Rotation error**: Components mixed or scaled wrong
   - PosAng calculation error
   - Wrong atan2 argument order

3. **Unit error**: Factor of 1000 or similar
   - mas vs arcsec confusion
   - Missing or double thetaE conversion

4. **Large systematic offset**: > 10% of signal
   - Reference frame mismatch
   - Wrong coordinate system

## Troubleshooting

### "Import bagle could not be resolved"

The validation scripts need BAGLE installed. Add to your Python path:

```bash
export PYTHONPATH="/Users/malpas.1/Code/BAGLE_Microlensing/src:$PYTHONPATH"
```

Or install BAGLE:
```bash
cd /Users/malpas.1/Code/BAGLE_Microlensing
pip install -e .
```

### "Missing parameters in gulls header"

The validation script needs specific parameters in the `.lc` file header.
See `BAGLE_VALIDATION_PLAN.md` Section "Step 1" for how to modify `outputLightcurve.cpp`.

### "RMS difference too large"

1. Check the **magnitude** of the difference:
   - < 1% of signal: probably OK (numerical differences)
   - > 10% of signal: likely a real problem

2. Check for **systematic offsets**:
   - Constant offset: reference frame issue
   - Inverted component: sign error
   - Scaled difference: unit error

3. Look at the **residual plots**:
   - Random scatter: numerical tolerance issue
   - Trend with time: proper motion issue
   - Trend with magnification: physics error

## Next Steps

1. **Run the simple validation**:
   ```bash
   python validation_bagle_simple.py
   ```
   - Understand what BAGLE predicts
   - Check that BAGLE is working correctly
   - Get reference values

2. **Modify gulls output** (if not already done):
   - Add parameters to header (see BAGLE_VALIDATION_PLAN.md)
   - Rebuild and run smoke tests
   - Check that headers are populated

3. **Run full validation**:
   ```bash
   python validation_bagle.py smoke_test/output/std/smoke_std_0_*.det.lc
   ```
   - Compare gulls to BAGLE
   - Analyze differences
   - Fix any issues

4. **Iterate**:
   - Adjust tolerances if needed
   - Fix coordinate transformations
   - Re-run validation
   - Document results

5. **Phase 2: Absolute astrometry**:
   - Validate (RA, Dec) outputs
   - Check proper motion application
   - Verify cos(Dec) factors
   - Compare to BAGLE's `get_astrometry()`

## References

- **BAGLE Documentation**: `/Users/malpas.1/Code/BAGLE_Microlensing/docs/`
- **BAGLE Tutorials**: 
  - `BAGLE_TUTORIAL.txt`
  - `model_fit_tutorial.txt`
  - `Frame_conversion_examples.txt`
- **VBMicrolensing**: `/Users/malpas.1/Code/VBMicrolensing/`
- **Gulls Implementation**: 
  - `src/pllxLightcurveGenerator.cpp` (astrometry calculation)
  - `src/outputLightcurve.cpp` (output formatting)

## Contact

If you run into issues or have questions about the validation:

1. Check `BAGLE_VALIDATION_PLAN.md` for detailed explanations
2. Look at the reference plots from `validation_bagle_simple.py`
3. Compare your gulls output format to BAGLE's expectations
4. Check coordinate convention documentation in BAGLE

Remember: We expect some differences due to L2 vs JPL Horizons and numerical methods.
The goal is to ensure the physics is correct, not to achieve perfect numerical agreement.
