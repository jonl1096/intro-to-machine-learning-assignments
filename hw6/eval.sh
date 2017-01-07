#!/usr/bin/env bash
for D in ./data/*
do
    currdir=$(basename $D)
    pref="./data/"
    echo "------ DATA SET: ${currdir} ---------------"
    echo "MC_PERCEPTRON"
    python ./code/classify.py --mode train --algorithm mc_perceptron --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train --online-training-iterations 5
    python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
done