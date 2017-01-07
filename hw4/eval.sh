for D in ./data/*
do
    currdir=$(basename $D)
    pref="./data/"
    echo "------ DATA SET: ${currdir} ---------------"
    # echo "NORMAL"
    # python ./code/classify.py --mode train --algorithm perceptron --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train 
    # python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    # python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    # echo "AVERAGE"
    # python ./code/classify.py --mode train --algorithm averaged_perceptron --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train
    # python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    # python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    # echo "MARGIN"
    # python ./code/classify.py --mode train --algorithm margin_perceptron --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train
    # python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    # python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    # echo "PEGASOS"
    # python ./code/classify.py --mode train --algorithm pegasos --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train
    # python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    # python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    # echo "KNN"
    # python ./code/classify.py --mode train --algorithm knn --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train --knn 5
    # python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    # python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    # echo "DISTANCE_KNN"
    # python ./code/classify.py --mode train --algorithm distance_knn --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train --knn 5
    # python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    # python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    # echo "ADABOOST"
    # python ./code/classify.py --mode train --algorithm adaboost --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train --num-boosting-iterations 10
    # python ./code/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    # python compute_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    echo "LAMBDA_MEANS"
    python ./code/classify.py --mode train --algorithm lambda_means --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train --cluster-lambda 0.0 --clustering-training-iterations 10
    python ./code/classify.py --mode test --algorithm lambda_means --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    python cluster_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    python number_clusters.py "$pref$currdir/$currdir".predictions
    # echo "NB_CLUSTERING"
    # python ./code/classify.py --mode train --algorithm nb_clustering --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train --num-clusters 3 --clustering-training-iterations 10
    # python ./code/classify.py --mode test --algorithm nb_clustering --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".dev  --predictions-file "$pref$currdir/$currdir".predictions
    # python cluster_accuracy.py "$pref$currdir/$currdir".dev "$pref$currdir/$currdir".predictions
    # python number_clusters.py "$pref$currdir/$currdir".predictions
done