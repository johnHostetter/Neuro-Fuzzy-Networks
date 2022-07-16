#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:06:21 2021

@author: john

This code uses the following notation:
    'n' is the number of input-dimensions. In the LazyPOP paper, this value is represented with 'n1'.
    'm' is the number of output-dimensions. In the LazyPOP paper, this value is represented with 'n5'.
    
There are some references to X^{(r)} or Y^{(r)}; this means the r'th training observation's input data 
or output data, respectively.
"""

import numpy as np

def noise_criterion(X, Y, input_cluster_membership_functions, output_cluster_membership_functions, NC):
    """
    The Noise Criterion NC is used in the proposed LazyPOP learning algorithm to detect noisy training data.
    
    A noise threshold can be defined such that a training data sample (X^{(r), Y^{(r)}}) with subliminal
    minimum membership are pruned away as noise.
    
    This is described in equation (e.3.1)
    
        min{min_{a}{\mu_{a} {(X^{(r)})} , \mu_{b} {(Y^{(r)})} }} < NC   (e.3.1)
        
    where \mu_{a} is the membership function representing cluster in the input space;
    where \mu_{b} is the membership function representing cluster in the output space;
    
    This threshold is called the Noise Criterion NC. It scrutinizes the training data to eradicate noisy ones.
    
    The remanant set of scrupulous training data, facilitate the generation of legitimate fuzzy rules.

    Parameters
    ----------
    X : iterable such as a list, tuple, or Numpy array
        A n-tuple representing a single training observation, where n is the number of input-dimensions.
    Y : iterable such as a list, tuple, or Numpy array
        A m-tuple representing a single training observation, where m is the number of output-dimensions.
    input_cluster_membership_functions : iterable such as a list, tuple, or Numpy array
        A list containing the cluster membership functions found for the input space.
    output_cluster_membership_functions : iterable such as a list, tuple, or Numpy array
        A list containing the cluster membership functions found for the output space.
    NC : float
        A user-defined threshold such that any training observation that achieves a value below this becomes discarded.

    Returns
    -------
    boolean
        Returns True if the training observation is determined to be noise and should be discarded. False otherwise.

    """
    input_cluster_membership_functions_degrees = []
    output_cluster_membership_functions_degrees = []
    for input_cluster_membership_function in input_cluster_membership_functions:
        input_cluster_membership_functions_degrees.append(input_cluster_membership_function(X))
    for output_cluster_membership_function in output_cluster_membership_functions:
        output_cluster_membership_functions_degrees.append(output_cluster_membership_function(Y))
    # minimum_cluster_membership_degree = min([np.max(input_cluster_membership_functions_degrees), np.max(output_cluster_membership_functions_degrees)])
    minimum_cluster_membership_degree = min(np.median(input_cluster_membership_functions_degrees), np.median(output_cluster_membership_functions_degrees))

    return minimum_cluster_membership_degree < NC

def ambiguity_criterion(X, Y, input_cluster_membership_functions, output_cluster_membership_functions, AC):
    """
    The Ambiguity Criterion AC is used to cull ambiguous training data.

    Ambiguous training data are produced when the system is in some nebulous states. Under such predicament, the behaviors
    of the system are so dubious that it is impossible to associate them with particular system process. Equivocal training
    data will manifest themselves as points in the intersection between clusters.

    If one uses fuzzy membership functions \mu_{a_1}, \mu_{a_2} (or \mu_{b_1}, \mu_{b_2}) to express two different clusters
    in the input (or output) space, then those training data that satisfy the equation (e.3.2.) are considered as ambiguous data.
    
    They will lie within the interval E as shown in Figure 4 (consult the cited paper).
    
        min{ 
            min_{ a_{1} != a_{2} } || \mu_{a_{1}}(X^{(r)}) - \mu_{a_{2}}(X^{(r)}) || ,    (e.3.2)
            min_{ b_{1} != b_{2} } || \mu_{b_{1}}(Y^{(r)}) - \mu_{b_{2}}(Y^{(r)}) || , 
            }
        
    Ambiguous training data contribute simultaneously to more than one fuzzy rule and are not exemplary. Hence, their removal
    will boost the performance of the learning algorithm.

    Parameters
    ----------
    X : iterable such as a list, tuple, or Numpy array
        A n-tuple representing a single training observation, where n is the number of input-dimensions.
    Y : iterable such as a list, tuple, or Numpy array
        A m-tuple representing a single training observation, where m is the number of output-dimensions.
    input_cluster_membership_functions : iterable such as a list, tuple, or Numpy array
        A list containing the cluster membership functions found for the input space.
    output_cluster_membership_functions : iterable such as a list, tuple, or Numpy array
        A list containing the cluster membership functions found for the output space.
    AC : float
        A user-defined threshold such that any training observation that achieves a value below this becomes discarded.

    Returns
    -------
    boolean
        Returns True if the training observation is determined to be ambiguous and should be discarded. False otherwise.

    """
    input_cluster_membership_functions_degrees = []
    output_cluster_membership_functions_degrees = []
    for outer_idx, a1 in enumerate(input_cluster_membership_functions):
        for inner_idx, a2 in enumerate(input_cluster_membership_functions):
            if outer_idx != inner_idx:
                a1_degree = a1(X)
                a2_degree = a2(X)
                raise NotImplementedError()
    for outer_idx, b1 in enumerate(output_cluster_membership_functions):
        for inner_idx, b2 in enumerate(output_cluster_membership_functions):
            if outer_idx != inner_idx:
                b1_degree = b1(X)
                b2_degree = b2(X)
                raise NotImplementedError()
    minimum_cluster_membership_degree = np.min(np.min(input_cluster_membership_functions_degrees), np.min(output_cluster_membership_functions_degrees))
    
    return minimum_cluster_membership_degree < AC