#!/bin/bash
# Test if debug build (which may zero-initialize memory) behaves differently

echo "=========================================="
echo "Debug vs Release Build Test"
echo "=========================================="
echo ""

SEED=12345
PARAMFILE="bug_replication_test/test_houston.prm"
OUTDIR="bug_replication_test/output"

# Ensure seed is set
sed -i.bak "s/RANDOM_SEED=.*/RANDOM_SEED=$SEED/" $PARAMFILE

echo "Building RELEASE version (current build)..."
cd /Users/malpas.1/Code/gulls_mp
cmake --build build --config Release > /dev/null 2>&1

echo "Testing RELEASE build with seed $SEED..."
rm -f $OUTDIR/*.lc $OUTDIR/*.log
timeout 60s bin/gullsFish.x $PARAMFILE 0 > /dev/null 2>&1
RELEASE_EXIT=$?

if [ -f "$OUTDIR/bug_test_houston_0_0_0.all.lc" ]; then
    RELEASE_RESULT="✅ SUCCESS"
elif [ $RELEASE_EXIT -eq 124 ]; then
    RELEASE_RESULT="❌ TIMEOUT"
else
    RELEASE_RESULT="⚠️  FAILED"
fi

echo "  Release build: $RELEASE_RESULT"
echo ""

echo "Building DEBUG version..."
cmake -DCMAKE_BUILD_TYPE=Debug -S . -B build_debug > /dev/null 2>&1
cmake --build build_debug > /dev/null 2>&1

if [ -f "build_debug/bin/gullsFish.x" ]; then
    echo "Testing DEBUG build with seed $SEED..."
    rm -f $OUTDIR/*.lc $OUTDIR/*.log
    timeout 60s build_debug/bin/gullsFish.x $PARAMFILE 0 > /dev/null 2>&1
    DEBUG_EXIT=$?
    
    if [ -f "$OUTDIR/bug_test_houston_0_0_0.all.lc" ]; then
        DEBUG_RESULT="✅ SUCCESS"
    elif [ $DEBUG_EXIT -eq 124 ]; then
        DEBUG_RESULT="❌ TIMEOUT"
    else
        DEBUG_RESULT="⚠️  FAILED"
    fi
    
    echo "  Debug build:   $DEBUG_RESULT"
    echo ""
    
    if [ "$RELEASE_RESULT" != "$DEBUG_RESULT" ]; then
        echo "🎯 PROOF: Different results between debug and release!"
        echo "   This confirms the bug is related to uninitialized memory."
        echo "   Debug builds often zero-initialize, hiding the bug."
    fi
else
    echo "⚠️  Debug build failed to compile"
fi

# Restore original parameter file
mv ${PARAMFILE}.bak $PARAMFILE

echo ""
echo "=========================================="

