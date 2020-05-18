I implemented the k-NN classification algorithm and test it on the Iris dataset.
Program is written in Lec1.py file. Steps of the algorithm are given below.
I used Iris dataset.
Training and Test Sets In the Iris dataset, each flower has 50 samples. Place the first 30 samples from each flower class into the training set and put the rest of the samples into the test set.

Iris_Test.csv
Iris_Train.csv

I apply k-NN algorithm to classify test samples. I try different k values.

The results are shown below :

Euclid	    k=1	    Accuracy %: 96.67	 Error Count: 58/60
Manhattan	k=1	    Accuracy %: 96.67	 Error Count: 58/60
Euclid	    k=3	    Accuracy %: 98.33	 Error Count: 59/60
Manhattan	k=3	    Accuracy %: 96.67	 Error Count: 58/60
Euclid	    k=5	    Accuracy %: 98.33	 Error Count: 59/60
Manhattan	k=5	    Accuracy %: 95.0	 Error Count: 57/60
Euclid	    k=7	    Accuracy %: 96.67	 Error Count: 58/60
Manhattan	k=7	    Accuracy %: 95.0	 Error Count: 57/60
Euclid	    k=9	    Accuracy %: 96.67	 Error Count: 58/60
Manhattan	k=9	    Accuracy %: 96.67	 Error Count: 58/60
Euclid	    k=11	Accuracy %: 95.0	 Error Count: 57/60
Manhattan	k=11	Accuracy %: 95.0	 Error Count: 57/60
Euclid	    k=15	Accuracy %: 95.0	 Error Count: 57/60
Manhattan	k=15	Accuracy %: 95.0	 Error Count: 57/60

