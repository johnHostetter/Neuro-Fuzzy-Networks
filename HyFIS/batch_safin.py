#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:07:44 2021

@author: john
"""

import sys
import numpy as np

def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))

class SaFIN:
    """
        Layer 1 consists of the input (variable) nodes.
        Layer 2 is the antecedent nodes.
        Layer 3 is the rule nodes.
        Layer 4 consists of the consequent ndoes.
        Layer 5 is the output (variable) nodes.
        
        In the SaFIN model, the input vector is denoted as:
            x = (x_1, ..., x_p, ..., x_P)
            
        The corresponding desired output vector is denoted as:
            d = (d_1, ..., d_q, ..., d_Q),
            
        while the computed output is denoted as:
            y = (y_1, ..., y_q, ..., y_Q)
        
        The notations used are the following:
        
        $P$: number of input dimensions
        $Q$: number of output dimensions
        $I_{p}$: $p$th input node
        $O_{q}$: $q$th output node
        $J_{p}$: number of fuzzy clusters in $I_{p}$
        $L_{q}$: number of fuzzy clusters in $O_{q}$
        $A_{j_p}$: $j$th antecedent fuzzy cluster in $I_{p}$
        $C_{l_q}$: $l$th consequent fuzzy cluster in $O_{q}$
        $K$: number of fuzzy rules
        $R_{k}$: $k$th fuzzy rule
    """
    def __init__(self, term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule):
        """
        
        Parameters
        ----------
        term_dict : TYPE
            DESCRIPTION.
        antecedents_indices_for_each_rule : TYPE
            DESCRIPTION.
        consequents_indices_for_each_rule : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.term_dict = term_dict
        self.antecedents_indices_for_each_rule = antecedents_indices_for_each_rule
        self.consequents_indices_for_each_rule = consequents_indices_for_each_rule
                
        self.P = self.antecedents_indices_for_each_rule.shape[1]
        # temporary fix until MIMO
        self.consequents_indices_for_each_rule = np.reshape(consequents_indices_for_each_rule, (len(consequents_indices_for_each_rule), 1))
        self.Q = self.consequents_indices_for_each_rule.shape[1]
        self.K = self.consequents_indices_for_each_rule.shape[0]
        
        self.J = {}
        self.total_antecedents = 0
        for p in range(self.P):
            fuzzy_clusters_in_I_p = set(self.antecedents_indices_for_each_rule[:,p])
            self.J[p] = len(fuzzy_clusters_in_I_p)
            self.total_antecedents += self.J[p]
        
        # between inputs and antecedents
        self.W_1 = np.zeros((self.P, self.total_antecedents))
        start_idx = 0
        for p in range(self.P):
            end_idx = start_idx + self.J[p]
            self.W_1[p, start_idx:end_idx] = 1
            start_idx = end_idx
            # print(W_1[p])
        
        # between antecedents and rules
        self.W_2 = np.empty((self.total_antecedents, self.K))
        self.W_2[:] = np.nan
        for rule_index, antecedents_indices_for_rule in enumerate(self.antecedents_indices_for_each_rule):
            start_idx = 0
            for input_index, antecedent_index in enumerate(antecedents_indices_for_rule):
                self.W_2[start_idx + antecedent_index, rule_index] = 1
                start_idx += self.J[input_index]
        
    def input_layer(self, x):
        # where x is the input vector and x[i] or x_i would be the i'th element of that input vector
        # restructure the input vector into a matrix to make the condtion layer's calculations easier
        self.f1 = x
        return self.f1 # following paper implementation
    def condition_layer(self, o1):        
        activations = np.dot(o1, self.W_1) # the shape is (num of inputs, num of all antecedents)
        # activations = (self.W_1.T * o1).T # the shape is (num of inputs, num of all antecedents)
        # flat_activations = np.sum(activations, axis=0)
        
        flat_centers = self.term_dict['antecedent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['antecedent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values

        self.f2 = np.exp(-1.0 * (np.power(activations - flat_centers, 2) / np.power(flat_widths, 2)))
        
        return self.f2 # shape is (num of inputs, num of all antecedents)
    def rule_base_layer(self, o2):         
        # the following assertion is not necessarily true                  
        # if np.nansum(self.W_2) != (self.P * self.K):
        #     raise Exception('The antecedents for rules have not been properly assigned.')
        # self.f3 = np.nanmin(np.dot(o2, self.W_2), axis=1)
        
        rule_activations = np.swapaxes(np.multiply(o2, self.W_2.T[:, np.newaxis]), 0, 1) # the shape is (num of observations, num of rules, num of antecedents)
        self.f3 = np.nanmin(rule_activations, axis=2) # the shape is (num of observations, num of rules)
        return self.f3
    def consequence_layer(self, o3):
        self.L = {}
        self.total_consequents = 0
        for q in range(self.Q):
            fuzzy_clusters_in_O_q = set(self.consequents_indices_for_each_rule[:,q])
            self.L[q] = len(fuzzy_clusters_in_O_q)
            self.total_consequents += self.L[q]
        
        # between rules and consequents
        self.W_3 = np.empty((self.K, self.total_consequents))
        self.W_3[:] = np.nan
        for rule_index, consequent_indices_for_rule in enumerate(self.consequents_indices_for_each_rule):
            start_idx = 0
            for output_index, consequent_index in enumerate(consequent_indices_for_rule):
                self.W_3[rule_index, start_idx + consequent_index] = 1
                start_idx += self.L[output_index]
                
        consequent_activations = np.swapaxes(np.multiply(o3, self.W_3.T[:, np.newaxis]), 0, 1)
        self.f4 = np.nanmax(consequent_activations, axis=2)
        return self.f4
    def output_layer(self, o4):
        numerator = np.nansum((o4 * self.term_dict['consequent_centers'] * self.term_dict['consequent_widths']), axis=1)
        denominator = np.nansum((o4 * self.term_dict['consequent_widths']), axis=1)
        self.f5 = numerator / denominator
        return self.f5
        
        # shape = (o4.shape[0], )
        # f = np.zeros(shape)
        # # the following is SaFIN
        # consequents_indices = set(self.consequents_indices_for_each_rule[:,0])
        # numerator = 0.0
        # denominator = 0.0
        # for q in consequents_indices:
        #     numerator += (o4[0][q] * self.term_dict['consequent_centers'][0][q] * self.term_dict['consequent_widths'][0][q])
        #     denominator += (o4[0][q] * self.term_dict['consequent_widths'][0][q])
        # return numerator / denominator
    def feedforward(self, x):
        self.o1 = self.input_layer(x)
        self.o2 = self.condition_layer(self.o1)
        self.o3 = self.rule_base_layer(self.o2)
        self.o4 = self.consequence_layer(self.o3)
        self.o5 = self.output_layer(self.o4)
        return self.o5
    def backpropagation(self, x, y):
        
        # (1) calculating the error signal in the output layer
        
        e5_m = -1.0 * (y - self.o5) # y actual minus y predicted
        
        # delta widths
        
        shape = self.term_dict['consequent_widths'].shape
        consequent_delta_widths = np.empty(shape)
        consequent_delta_widths[:] = np.nan
        
        lhs_numerator = np.nansum(self.o4 * self.term_dict['consequent_widths'])
        rhs_numerator = np.nansum(self.o4 * self.term_dict['consequent_widths'] * self.term_dict['consequent_centers'])
        
        l = 0 # modify this in order to work beyond MISO
        for k in range(shape[1]):
            consequent_delta_widths[l][k] = self.o4[l, k] * ((self.term_dict['consequent_centers'][l, k] * lhs_numerator) - rhs_numerator)
            consequent_delta_widths[l][k] /= np.power(lhs_numerator, 2)
        
        consequent_delta_widths *= e5_m
        
        # delta centers
        
        shape = self.term_dict['consequent_centers'].shape
        consequent_delta_centers = np.empty(shape)
        consequent_delta_centers[:] = np.nan
        
        l = 0
        for k in range(shape[1]):
            consequent_delta_centers[l, k] = self.term_dict['consequent_widths'][l, k] * self.term_dict['consequent_centers'][l, k]
            consequent_delta_centers[l, k] /= np.nansum(self.o4 * self.term_dict['consequent_widths'])
        
        consequent_delta_centers *= e5_m
        
        # calculate the error in layer 4
        
        shape = self.o4.shape
        dy5_dy4k = np.empty(shape)
        dy5_dy4k[:] = np.nan
        
        # same calculations as earlier, but repeated to avoid confusion
        # lhs_numerator = np.nansum(self.o4 * self.term_dict['consequent_widths'])
        # rhs_numerator = np.nansum(self.o4 * self.term_dict['consequent_widths'] * self.term_dict['consequent_centers'])
        
        l = 0 # modify this in order to work beyond MISO
        for k in range(shape[1]):
            dy5_dy4k[l][k] = self.term_dict['consequent_widths'][l, k] * ((self.term_dict['consequent_centers'][l, k] * lhs_numerator) - rhs_numerator)
            dy5_dy4k[l][k] /= np.power(lhs_numerator, 2)
            
        e4_m = e5_m * dy5_dy4k
        
        # calculate the error in layer 3
        
        shape = self.o3.shape
        e3_m = np.empty(shape)
        e3_m[:] = np.nan
        
        m = 0
        for j in range(e3_m.shape[0]):
            k = self.consequents_indices_for_each_rule[j]
            e3_m[j] = e4_m[m][k]
                
        # calculate the error in layer 2
        
        # delta centers
        
        r_term = None
        for j in range(e3_m.shape[0]):
            I_j = self.antecedents_indices_for_each_rule[j] # the set of indices of the nodes in layer 2 that are connected to node j in layer 3 
            vals = []
            for idx, i in enumerate(I_j):
                vals.append(self.o2[idx, i])
            r = np.nanargmin(vals)
            r_term = I_j[r]
        
        
        shape = self.term_dict['antecedent_centers'].shape
        dE_dyi2 = np.zeros(shape)
        
        dE_dyi2[r][r_term] = np.nansum(e3_m)
        
        antecedent_delta_centers = dE_dyi2 * self.o2 * (2 * (self.o1 - self.term_dict['antecedent_centers'])) / np.power(self.term_dict['antecedent_widths'], 2)
        
        shape = self.term_dict['antecedent_centers'].shape
        dE_dyj2 = np.empty(shape)
        dE_dyj2[:] = np.nan
        
        # delta widths
        
        antecedent_delta_widths = dE_dyi2 * self.o2 * ((2 * np.power(self.o1 - self.term_dict['antecedent_centers'], 2)) / np.power(self.term_dict['antecedent_widths'], 3))
                
        return consequent_delta_centers, consequent_delta_widths, antecedent_delta_centers, antecedent_delta_widths
        
        # delta centers
        shape = self.term_dict['consequent_centers'].shape
        consequent_delta_c = np.empty(shape)
        consequent_delta_c[:] = np.nan
        numerator = (self.o4 / self.term_dict['consequent_widths'])
        denominator = (np.nansum(self.o4 / self.term_dict['consequent_widths']))
        if np.isnan(e5_m * (numerator / denominator)).all():
            print('line 110 %s * (%s / %s)' % (e5_m, numerator, denominator))
            sys.exit()
        consequent_delta_c = e5_m * (numerator / denominator)
                
        # delta widths
        shape = self.term_dict['consequent_widths'].shape
        consequent_delta_widths = np.empty(shape)
        consequent_delta_widths[:] = np.nan
        numerator_lhs = np.nansum(self.term_dict['consequent_centers'] * self.o4 / self.term_dict['consequent_widths']) * self.o4 * pow(self.term_dict['consequent_widths'], -2)
        numerator_rhs = np.nansum(self.o4 / self.term_dict['consequent_widths']) * self.term_dict['consequent_centers'] * self.o4 * pow(self.term_dict['consequent_widths'], -2)
        denominator = pow(np.nansum(self.o4 / self.term_dict['consequent_widths']), 2)
        if np.isnan((numerator_lhs - numerator_rhs) / denominator).all():
            print('%s / %s' % ((numerator_lhs - numerator_rhs), denominator))
            sys.exit()
        consequent_delta_widths = (numerator_lhs - numerator_rhs) / denominator
        
        # (2) calculating the error signal in the consequence layer
        
        numerator_lhs = (self.term_dict['consequent_centers'] / self.term_dict['consequent_widths']) * (np.nansum(self.o4 / self.term_dict['consequent_widths']))
        numerator_rhs = (np.nansum(self.term_dict['consequent_centers'] * self.o4 / self.term_dict['consequent_widths'])) / self.term_dict['consequent_widths']
        denominator = pow(np.nansum(self.o4 / self.term_dict['consequent_widths']), 2)
        
        e4_m = e5_m * ((numerator_lhs - numerator_rhs) / denominator)
        
        # (3) calculating the error signal in the rule-base layer
        
        shape = self.consequents_indices_for_each_rule.shape
        e3 = np.zeros(shape)
        for l in set(self.consequents_indices_for_each_rule):
            relevant_rule_indices = np.where(self.consequents_indices_for_each_rule == l)[0]
            m = 0 # iterate over m's if this is to be redesigned to work on multi-dimensional output
            e3[relevant_rule_indices] = np.sum(e4_m[m, l])
        
        # (4) calculating the error signal in the condition layer

        shape = self.term_dict['antecedent_centers'].shape
        e2 = np.zeros(shape) # requires i, j to index
        q = np.zeros(e3.shape)
        for k, antecedent_indices_for_rule in enumerate(self.antecedents_indices_for_each_rule):
            for i, j in enumerate(antecedent_indices_for_rule):
                if self.o2[i, j] == self.f3[k]:
                    q[k] = e3[k]
    
        for i in range(self.antecedents_indices_for_each_rule.shape[1]):
            col = self.antecedents_indices_for_each_rule[:,i]
            for j in set(col):
                relevant_rule_indices = np.where(col == j)[0]
                e2[i, j] = np.nansum(q[relevant_rule_indices])
                
        # delta centers
        shape = self.term_dict['antecedent_centers'].shape
        antecedent_delta_c = np.empty(shape)
        antecedent_delta_c[:] = np.nan
        # print('line 163 %s' % pow(self.term_dict['antecedent_widths'], 2))
        antecedent_delta_c = (e2 * self.o2) * ((2 * (self.o1 - self.o2)) / pow(self.term_dict['antecedent_widths'], 2))
        
        # delta widths
        shape = self.term_dict['antecedent_widths'].shape
        antecedent_delta_widths = np.empty(shape)
        antecedent_delta_widths[:] = np.nan
        # print('line 170 %s' % pow(self.term_dict['antecedent_widths'], 3))
        antecedent_delta_widths = (e2 * self.o2) * ((self.o1 - self.o2) / pow(self.term_dict['antecedent_widths'], 3))
                
        return consequent_delta_c, consequent_delta_widths, antecedent_delta_c, antecedent_delta_widths