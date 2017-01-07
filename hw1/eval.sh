for D in ./data/*
do
    currdir=$(basename $D)
    pref="./data/"
    echo "------ DATA SET: ${currdir} ---------------"
    echo "NORMAL"
    python ./code/classify.py --mode train --algorithm perceptron --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train
    python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
#    echo "AVERAGE"
#    python ./code/classify.py --mode train --algorithm averaged_perceptron --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train
#        python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
#        python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
done