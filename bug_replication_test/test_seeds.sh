#!/bin/bash
# Test multiple random seeds with high timeout to replicate Matt's conditions
# (Matt didn't have LC_TIMEOUT, so lucky runs would complete, unlucky ones would hang)

echo "=========================================="
echo "Serendipity Test - Multiple Random Seeds"
echo "=========================================="
echo "Testing with 600s timeout (vs Matt's no timeout)"
echo "Hypothesis: Some seeds complete successfully (lucky thE value)"
echo "            Some seeds timeout (unlucky thE value -> inf)"
echo ""

# Test a variety of seeds
SEEDS=(12345 54321 99999 42 7777 1 88888 33333 66666 11111)

PARAMFILE="bug_replication_test/test_no_timeout.prm"
OUTDIR="bug_replication_test/output/bug_test_no_timeout"
mkdir -p $OUTDIR

RESULTS_FILE="bug_replication_test/seed_results.csv"
echo "Seed,Status,Time_sec,LC_File_Exists,Log_Has_Inf" > $RESULTS_FILE

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Testing SEED=$SEED"
    echo "=========================================="
    
    # Update seed in parameter file
    sed "s/RANDOM_SEED=.*/RANDOM_SEED=$SEED/" $PARAMFILE > ${PARAMFILE}.tmp
    mv ${PARAMFILE}.tmp $PARAMFILE
    
    # Clean previous output
    rm -f $OUTDIR/*.lc $OUTDIR/*.log $OUTDIR/*.occ_data
    
    # Run with time measurement
    START=$(date +%s)
    ./bin/gullsFish.x -i $PARAMFILE -s 0 > /dev/null 2>&1
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    # Check results
    LC_EXISTS="No"
    HAS_INF="Unknown"
    
    if [ -f "$OUTDIR/bug_test_no_timeout_0_0_0.all.lc" ]; then
        LC_EXISTS="Yes"
        STATUS="✅ SUCCESS"
    else
        LC_EXISTS="No"
    fi
    
    # Check log for inf values
    if [ -f "$OUTDIR/bug_test_no_timeout_0_0.log" ]; then
        if grep -q "inf" $OUTDIR/bug_test_no_timeout_0_0.log; then
            HAS_INF="Yes"
            if [ "$LC_EXISTS" = "No" ]; then
                STATUS="❌ TIMEOUT (inf detected)"
            else
                STATUS="⚠️  SUCCESS but had inf"
            fi
        else
            HAS_INF="No"
            if [ "$LC_EXISTS" = "Yes" ]; then
                STATUS="✅ SUCCESS (clean)"
            else
                STATUS="❌ FAILED (other error)"
            fi
        fi
    fi
    
    echo "  Result: $STATUS (${ELAPSED}s)"
    echo "  LC file: $LC_EXISTS, Has inf: $HAS_INF"
    echo "$SEED,$STATUS,$ELAPSED,$LC_EXISTS,$HAS_INF" >> $RESULTS_FILE
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
cat $RESULTS_FILE
echo ""

SUCCESSES=$(grep "SUCCESS" $RESULTS_FILE | wc -l)
TIMEOUTS=$(grep "TIMEOUT" $RESULTS_FILE | wc -l)

echo "Results:"
echo "  Clean successes: $SUCCESSES"
echo "  Timeouts (inf):  $TIMEOUTS"
echo ""

if [ $SUCCESSES -gt 0 ] && [ $TIMEOUTS -gt 0 ]; then
    echo "🎯 PROOF: Both successes AND timeouts observed!"
    echo "   This proves Matt's success was SERENDIPITOUS (lucky random seed)"
    echo "   Different seeds produce different results due to uninitialized memory"
elif [ $TIMEOUTS -gt 5 ]; then
    echo "⚠️  Mostly/all timeouts - bug is consistent"
    echo "   (This actually makes it easier to prove it's a bug)"
elif [ $SUCCESSES -gt 5 ]; then
    echo "⚠️  Mostly/all successes - lucky compiler behavior?"
    echo "   May need to test release vs debug builds"
fi

