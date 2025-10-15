#!/bin/bash
# Test script to demonstrate probabilistic nature of uninitialized memory bug

echo "=========================================="
echo "Bug Replication Test - Uninitialized thE"
echo "=========================================="
echo ""
echo "Testing Houston catalog with multiple random seeds"
echo "Hypothesis: Some seeds work (lucky), some timeout (unlucky)"
echo ""

# Test seeds - variety to increase chance of showing both behaviors
SEEDS=(12345 54321 11111 99999 42 7777 13579 24680 1 100000)

# Output directory
OUTDIR="bug_replication_test/output"
PARAMFILE="bug_replication_test/test_houston.prm"

# Results tracking
RESULTS_FILE="bug_replication_test/results.txt"
echo "Seed,Status,Source2_s,Source2_rho,Time" > $RESULTS_FILE

echo "Running 10 tests with different random seeds..."
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Test with RANDOM_SEED=$SEED"
    echo "----------------------------------------"
    
    # Clean previous output
    rm -f $OUTDIR/*.lc $OUTDIR/*.log $OUTDIR/*.occ_data
    
    # Update parameter file with new seed
    sed -i.bak "s/RANDOM_SEED=.*/RANDOM_SEED=$SEED/" $PARAMFILE
    
    # Run simulation with timeout
    START_TIME=$(date +%s)
    timeout 60s ./bin/gullsFish.x -i $PARAMFILE -s 0 > /dev/null 2>&1
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    # Check results
    if [ $EXIT_CODE -eq 124 ]; then
        echo "  ❌ TIMEOUT after ${ELAPSED}s (likely inf in scomp_s)"
        echo "$SEED,TIMEOUT,inf,inf,$ELAPSED" >> $RESULTS_FILE
    elif [ -f "$OUTDIR/bug_test_houston_0_0_0.all.lc" ]; then
        # Extract Source2_s and Source2_rho from log
        LOG_FILE="$OUTDIR/bug_test_houston_0_0_0.log"
        if [ -f "$LOG_FILE" ]; then
            SOURCE2_S=$(grep "Source2_s" $LOG_FILE | awk '{print $2}' || echo "N/A")
            SOURCE2_RHO=$(grep "Source2_rho" $LOG_FILE | awk '{print $2}' || echo "N/A")
        else
            SOURCE2_S="N/A"
            SOURCE2_RHO="N/A"
        fi
        echo "  ✅ SUCCESS in ${ELAPSED}s (Source2_s=$SOURCE2_S, Source2_rho=$SOURCE2_RHO)"
        echo "$SEED,SUCCESS,$SOURCE2_S,$SOURCE2_RHO,$ELAPSED" >> $RESULTS_FILE
    else
        echo "  ⚠️  FAILED (no output, exit code $EXIT_CODE)"
        echo "$SEED,FAILED,N/A,N/A,$ELAPSED" >> $RESULTS_FILE
    fi
    echo ""
done

# Restore original parameter file
mv ${PARAMFILE}.bak $PARAMFILE

echo "=========================================="
echo "Summary"
echo "=========================================="
cat $RESULTS_FILE
echo ""

# Count successes and timeouts
SUCCESSES=$(grep -c "SUCCESS" $RESULTS_FILE)
TIMEOUTS=$(grep -c "TIMEOUT" $RESULTS_FILE)

echo "Results:"
echo "  Successes: $SUCCESSES / 10"
echo "  Timeouts:  $TIMEOUTS / 10"
echo ""

if [ $SUCCESSES -gt 0 ] && [ $TIMEOUTS -gt 0 ]; then
    echo "🎯 PROOF OF BUG: Both successes and timeouts observed!"
    echo "   This demonstrates the probabilistic nature of uninitialized memory."
    echo "   Matt's tests succeeded by LUCK, not because the code is correct."
elif [ $TIMEOUTS -eq 10 ]; then
    echo "⚠️  All tests timed out (100% failure rate)"
    echo "   This is consistent with the bug, but we need a 'lucky' run to prove it's probabilistic."
else
    echo "⚠️  All tests succeeded (unexpected)"
    echo "   Need to investigate further - may need different seeds or compiler flags."
fi

echo ""
echo "Full results saved to: $RESULTS_FILE"

