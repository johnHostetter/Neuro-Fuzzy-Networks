#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:39:32 2021

@author: john
"""

import sys
import numpy as np

def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))

class SaFIN:
    def __init__(self, term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule):
        self.term_dict = term_dict
        self.antecedents_indices_for_each_rule = antecedents_indices_for_each_rule
        self.consequents_indices_for_each_rule = consequents_indices_for_each_rule
    def input_layer(self, x):
        # where x is the input vector and x[i] or x_i would be the i'th element of that input vector
        # restructure the input vector into a matrix to make the condtion layer's calculations easier
        o1 = np.repeat(x[np.newaxis, ...], self.term_dict['antecedent_centers'].shape[1], axis=0).T # directly transmit the non-fuzzy input values to the second layer
        self.f1 = x
        return o1
    def condition_layer(self, o1):
        # the input to the conditon layer is o_1
        # print('line 26 %s' % pow((o1 - self.term_dict['antecedent_centers']), 2))
        # f = -1 * (pow((o1 - self.term_dict['antecedent_centers']), 2) / self.term_dict['antecedent_widths'])
        # self.f2 = f
        # o2 = np.exp(f)
        # return o2
        self.f2 = gaussian(o1, self.term_dict['antecedent_centers'], self.term_dict['antecedent_widths'])
        return self.f2
    def rule_base_layer(self, o2):
        f = np.ones(len(self.antecedents_indices_for_each_rule))
        for k, antecedents_indices_for_rule in enumerate(self.antecedents_indices_for_each_rule):
            for i, j in enumerate(antecedents_indices_for_rule):
                f[k] = min(f[k], o2[i, j])
        self.f3 = f
        o3 = f
        return o3
    def consequence_layer(self, o3):
        # this has to be redesigned to work on multi-dimensional output
        # shape = (max(self.consequents_indices_for_each_rule) + 1, 1)
        shape = self.term_dict['consequent_centers'].shape
        f = np.empty(shape)
        f[:] = np.nan
        # f = np.zeros((len(set(self.consequents_indices_for_each_rule)), 1))
        for l in set(self.consequents_indices_for_each_rule):
            relevant_rule_indices = np.where(self.consequents_indices_for_each_rule == l)[0]
            m = 0 # iterate over m's if this is to be redesigned to work on multi-dimensional output
            f[m, l] = np.nanmax(o3[relevant_rule_indices])
        self.f4 = f
        o4 = f 
        return o4
    def output_layer(self, o4):
        shape = (o4.shape[0], )
        f = np.zeros(shape)
        # the following is SaFIN
        consequents_indices = set(self.consequents_indices_for_each_rule)
        numerator = 0.0
        denominator = 0.0
        for q in consequents_indices:
            numerator += (o4[0][q] * self.term_dict['consequent_centers'][0][q] * self.term_dict['consequent_widths'][0][q])
            denominator += (o4[0][q] * self.term_dict['consequent_widths'][0][q])
        return numerator / denominator
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