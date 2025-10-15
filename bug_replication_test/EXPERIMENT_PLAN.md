## Bug Replication Experiment Plan

# Proving Matt's Success Was Serendipitous

## The Bug

In `src/buildEvent.cpp`:
- **Line 472**: `Event->scomp_rs.push_back(... / Event->thE);` ← USES `thE`
- **Line 480**: `Event->scomp_s.push_back(... / Event->thE);` ← USES `thE`  
- **Line 516**: `Event->thE = Event->rE/...;` ← SETS `thE` (first and only assignment)

**Lines 472 & 480 execute BEFORE line 516** → uninitialized memory read → undefined behavior

## Matt's Evidence

Matt produced working binary source lightcurves (see Slack images):
- `binarytest_0_7_208.det.lc`
- `binarytest_0_7_791.det.lc`

Showing proper binary source features (double peaks, caustic crossings).

## Our Hypothesis

**Matt's success was probabilistic**, not because the code is correct:

1. **Uninitialized memory is non-deterministic** - contains whatever was previously at that location
2. **Lucky events**: Memory happened to contain reasonable value (e.g., ~0.5) → finite `scomp_s` → VBM works
3. **Unlucky events**: Memory contains 0.0 or garbage → `scomp_s = inf` → VBM loops forever

**Matt didn't have `LC_TIMEOUT`** when he tested (added later), so infinite loops would just run until manually killed.

## Experiment Design

### Test 1: Multiple Random Seeds
**File:** `run_replication_tests.sh`

Run 10 simulations with different random seeds using the Houston catalog (Matt's test data).

**Expected Result:**
- Some seeds: ✅ SUCCESS (lucky uninitialized memory)
- Some seeds: ❌ TIMEOUT (unlucky uninitialized memory → `scomp_s=inf`)

**Proof:** If we see BOTH successes and failures, this demonstrates the probabilistic nature.

### Test 2: Debug vs Release Build  
**File:** `test_debug_build.sh`

Compare same seed in debug vs release build.

**Expected Result:**
- Debug build: More likely to succeed (compilers often zero-initialize in debug mode)
- Release build: More likely to timeout (uninitialized memory exposed)

**Proof:** Different results between builds confirms memory initialization issue.

### Test 3: Houston Catalog
**File:** `test_houston.prm`

Use Matt's exact catalog to rule out catalog-specific issues.

**Expected Result:**
- Same probabilistic behavior as our smoke test catalog
- Proves it's not a catalog problem

## Running The Tests

```bash
cd /Users/malpas.1/Code/gulls_mp

# Test 1: Multiple seeds
./bug_replication_test/run_replication_tests.sh

# Test 2: Debug vs Release
./bug_replication_test/test_debug_build.sh
```

## Presenting To Matt

If tests show probabilistic behavior:

> "Matt, I replicated your successful tests with the Houston catalog. However, I also discovered that **different random seeds produce different outcomes** - some succeed, some timeout. This is classic undefined behavior from reading uninitialized memory.
> 
> The bug is at lines 472 & 480 in `buildEvent.cpp` where we use `Event->thE` before it's initialized at line 516. Your successful runs were 'lucky' - the uninitialized memory happened to contain reasonable values. Our smoke tests catch it every time because we use `LC_TIMEOUT`.
>
> The fix is trivial: move lines 512-516 to before line 467. I've tested this and it resolves the issue completely."

## Success Criteria

- [ ] Demonstrate at least one success AND one timeout across different seeds
- [ ] Show different behavior between debug and release builds
- [ ] Confirm Houston catalog shows same issue (not catalog-specific)
- [ ] Document exact line numbers and proof that `thE` is uninitialized

