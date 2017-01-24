# Double-Sampled-Multiclass-to-Binary-Reduction

Introduction: 

We address the problem of multiclass classification in the case where the number of classes is very large. We propose a multiclass to binary reduction strategy, in which we transform the original problem into a binary classification one over pairs of examples. We derive generalization bounds for the error of the classifier of pairs using local Rademacher complexity, and a double sampling strategy (in the terms of examples and classes) that speeds up the training phase while maintaining a very low memory usage. Experiments are carried for text classification on DMOZ and Wikipedia collections with up to 20,000 classes in order to show the efficiency of the proposed method.

Running Instructions: 

python3 run_script_m2b_github.py <train filename> <test filename> <example_samples> <class_sampling> <Candidates>

Where,
example_samples (mu): Number of examples to be taken per class ( e.g. values 1, 2, 5)
class_sampling: Sampling rate for choosing classes to sample ( e.g. 0.1, 0.01, 0.001) (Note: The minimum value for class_
sampling is set as 1 / Size of class, if user enters less than this value by default 1 class will be chosen.
Candidates (sigma): Number of candidate classes for prediction (e.g. 10, 20, 50)

Author Information: 
1.) Bikash Joshi
2.) Ioannis Partalas

License: This software is licensed under GNU General Public License version 3.0

Reference: https://arxiv.org/abs/1701.06511
