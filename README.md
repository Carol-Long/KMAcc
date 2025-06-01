# KMAcc
The official code of **Kernel Multiaccuracy** [Link to paper](https://drive.google.com/file/d/19TvapKLQ0y2gSHqKkNkKgXJrnCXB7Y9i/view?usp=sharing)

## Abstract
We demonstrate that recently introduced notions of multi-group fairness can be equivalently formulated as integral probability metrics (IPM). IPMs are the common information-theoretic tool that underly definitions such as multiaccuracy, multicalibration, and outcome indistinguishably. For multiaccuracy, this connection leads to a simple, yet powerful procedure for achieving multiaccuracy with respect to an infinite-dimensional class of functions defined by a reproducing kernel Hilbert space (RKHS): first perform a kernel regression of a model's errors, then subtract the resulting function from a model's predictions. We combine these results to develop a post-processing method that ensures multiaccuracy with respect to bounded-norm functions in an RKHS, enjoys provable performance guarantees, and, in binary classification benchmarks, achieves favorable  multiaccuracy  relative to competing methods. 
 
 ## Implementation Summary 
- class 'MAccWitness':  the class where witness function is defined. 
- class 'KMultiAcc': the class containing the main algorithm that ensures multiaccuracy using witness 
- function 'achieving_calibration_with_witness': the mega function that takes in a dataset and a baseline_model and returned an multi-group fair model
 
 ## Experiments
Experiments run on synthetic datasets and real-world datasets (Folktables, US census datasets) are included in separate jupyter notebooks.
