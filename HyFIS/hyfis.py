#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:42:36 2021

@author: john
"""

import numpy as np

from common import gaussian

def rule_creation(X, Y, antecedents, consequents, existing_rules=[], existing_weights=[]):
    rules = existing_rules
    weights = existing_weights
    for training_tuple in zip(X, Y):
        x = training_tuple[0]
        d = training_tuple[1]
        
        CF = 1.0 # certainty factor of this rule
        A_star_js = []
        for p in range(len(x)):
            SM_jps = []
            for j, A_jp in enumerate(antecedents[p]):
                SM_jp = gaussian(x[p], A_jp['center'], A_jp['sigma'])
                SM_jps.append(SM_jp)
            CF *= np.max(SM_jps)
            j_star_p = np.argmax(SM_jps)
            A_star_js.append(j_star_p)

        C_star_qs = []
        for q in range(len(d)):
            SM_jqs = []
            for j, C_jq in enumerate(consequents[q]):
                SM_jq = gaussian(d[q], C_jq['center'], C_jq['sigma'])
                SM_jqs.append(SM_jq)
            # print(SM_jqs)
            CF *= np.max(SM_jqs)
            j_star_q = np.argmax(SM_jqs)
            C_star_qs.append(j_star_q)
            
        R_star = {'A':A_star_js, 'C': C_star_qs, 'CF': CF}
        # print(R_star)
        
        if not rules:
            # no rules in knowledge base yet
            rules.append(R_star)
            weights.append(1.0)
        else:
            # check for uniqueness
            add_new_rule = True
            for k, rule in enumerate(rules):
                # print(k)
                try:
                    if (rule['A'] == R_star['A']) and (rule['C'] == R_star['C']):
                        # the generated rule is not unique, it already exists, enhance this rule's weight
                        weights[k] += 1.0
                        rule['CF'] = min(rule['CF'], R_star['CF'])
                        add_new_rule = False
                        break
                except ValueError: # this happens because R_star['A'] and R_star['C'] are Numpy arrays
                    if all(rule['A'] == list(R_star['A'])) and all(rule['C'] == list(R_star['C'])):
                        # the generated rule is not unique, it already exists, enhance this rule's weight
                        weights[k] += 1.0
                        rule['CF'] = min(rule['CF'], R_star['CF'])
                        add_new_rule = False
                        break
            if add_new_rule:
                rules.append(R_star)
                weights.append(1.0)
                
    # check for consistency
    all_antecedents = [rule['A'] for rule in rules]
    # unq, counts = np.unique(all_antecedents, axis=0, return_counts=True)
    # for k in np.where(counts > 1)[0]:
    #     # find the rules that match this
    #     repeated_antecedents = rules[k]['A']
    #     repeated_consequents = rules[k]['C']
    #     rule_indices = np.where((np.array(all_antecedents) == repeated_antecedents).all(axis=1))
    repeated_rule_indices = set()
    for k in range(len(rules)):
        indices = np.where(np.all(all_antecedents == np.array(rules[k]['A']), axis=1))[0]
        if len(indices) > 1: 
            if len(repeated_rule_indices) == 0:
                repeated_rule_indices.add(tuple(indices))
            # elif len(repeated_rule_indices) > 0 and indices not in np.unique(repeated_rule_indices, axis=1):
            elif len(repeated_rule_indices) > 0:
                repeated_rule_indices.add(tuple(indices))
    
    for indices in repeated_rule_indices:
        # weights_to_compare = [weights[idx] for idx in indices]
        weights_to_compare = [rules[idx]['CF'] for idx in indices]
        strongest_rule_index = indices[np.argmax(weights_to_compare)] # keep the rule with the greatest weight to it
        for index in indices:
            if index != strongest_rule_index:
                rules[index] = None
                weights[index] = None
    rules = [rules[k] for k, rule in enumerate(rules) if rules[k] is not None]
    weights = [weights[k] for k, weight in enumerate(weights) if weights[k] is not None]
    
    # testing using certainty factors on whether to keep based on this metric
    # weights = [weights[k] for k, weight in enumerate(weights) if rules[k]['CF'] >= 0.2]
    # rules = [rules[k] for k, rule in enumerate(rules) if rules[k]['CF'] >= 0.2]

    # need to check that no antecedent/consequent terms are "orphanned"
    
    all_antecedents = [rule['A'] for rule in rules]
    all_antecedents = np.array(all_antecedents)
    for p in range(len(x)):
        if len(antecedents[p]) == len(np.unique(all_antecedents[:,p])):
            continue
        else:
            # print('orphanned antecedent term') # need to implement this
            indices_for_antecedents_that_are_used = set(all_antecedents[:,p])
            updated_indices_to_map_to = list(range(len(indices_for_antecedents_that_are_used)))
            antecedents[p] = [antecedents[p][index] for index in indices_for_antecedents_that_are_used]
            
            paired_indices = list(zip(indices_for_antecedents_that_are_used, updated_indices_to_map_to))
            for index_pair in paired_indices: # the paired indices are sorted w.r.t. the original indices
                original_index = index_pair[0] # so, when we updated the original index to its new index
                new_index = index_pair[1] # we are guaranteed not to overwrite the last updated index
                all_antecedents[:,p][all_antecedents[:,p] == original_index] = new_index
            
    all_consequents = [rule['C'] for rule in rules]
    all_consequents = np.array(all_consequents)
    for q in range(len(d)):
        if len(consequents[q]) == len(np.unique(all_consequents[:,q])):
            continue
        else:
            # print('orphanned consequent term') # need to implement this
            indices_for_consequents_that_are_used = set(all_consequents[:,q])
            updated_indices_to_map_to = list(range(len(indices_for_consequents_that_are_used)))
            consequents[q] = [consequents[q][index] for index in indices_for_consequents_that_are_used]
            
            paired_indices = list(zip(indices_for_consequents_that_are_used, updated_indices_to_map_to))
            for index_pair in paired_indices: # the paired indices are sorted w.r.t. the original indices
                original_index = index_pair[0] # so, when we updated the original index to its new index
                new_index = index_pair[1] # we are guaranteed not to overwrite the last updated index
                all_consequents[:,q][all_consequents[:,q] == original_index] = new_index
                
    # update the rules in case any orphanned terms occurred
    for idx, rule in enumerate(rules):
        rule['A'] = all_antecedents[idx]
        rule['C'] = all_consequents[idx]

    return antecedents, consequents, rules, weights