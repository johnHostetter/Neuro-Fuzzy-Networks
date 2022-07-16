#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:39:43 2021

@author: john
"""

import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hyfis import rule_creation
from self_organizing import CLIP
from constant import PROBLEM_FEATURES
from common import gaussian, boolean_indexing
from lazypop import noise_criterion, ambiguity_criterion

from copy import deepcopy
from functools import partial
from sklearn.datasets import load_boston
from mlxtend.frequent_patterns import fpgrowth
from sklearn.neighbors import KNeighborsRegressor

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

SHOW_PLOTS = True

print('loading')

# data = pd.read_csv('./data/concat_data_with_critical.csv', delimiter=',')
data = pd.read_csv('./data/problem_level.csv', delimiter=',')
data = data[(data['decisionPoint'] == 'probStart')]

train_data = data[:10000]
test_data = data[10000:]

# output_features = ['normal Bellman Q value for action 0']
output_features = ['normal Bellman Q value for action 0', 
                    'normal Bellman Q value for action 1', 
                    'normal Bellman Q value for action 2']
ALL_FEATURES = deepcopy(PROBLEM_FEATURES)
ALL_FEATURES.extend(output_features)

# train_data[PROBLEM_FEATURES] = normalized_df=(train_data[PROBLEM_FEATURES]-train_data[PROBLEM_FEATURES].min())/(train_data[PROBLEM_FEATURES].max() - train_data[PROBLEM_FEATURES].min())
# test_data[PROBLEM_FEATURES] = normalized_df=(test_data[PROBLEM_FEATURES]-train_data[PROBLEM_FEATURES].min())/(train_data[PROBLEM_FEATURES].max()-train_data[PROBLEM_FEATURES].min())

print('normalization done')

# # feature importance for subsequent CLIP analysis
# from sklearn.ensemble import RandomForestRegressor
# X = data[PROBLEM_FEATURES]
# # Y = data['normal Bellman Q value for action 0'].values
# Y = data[output_features].values

# # define the model
# model = RandomForestRegressor()
# # fit the model
# model.fit(X, Y)
# # get importance
# importance = model.feature_importances_
# # summarize feature importance
# for i,v in enumerate(importance):
#  	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()
# # get the indices of the n largest elements from a Numpy array
# n = 12
# ind = np.argpartition(importance, -n)[-n:]
# selected_features = np.array(PROBLEM_FEATURES)[ind]
# selected_features = PROBLEM_FEATURES[:4]

selected_features = ['totalTimeStepsHint', 'nTellKCSession', 'stepsSinceLastHint',
       'nTellSession', 'problemDifficuly', 'pctElicitKC',
       'pctCorrectDeMorAll', 'pctElicit', 'nDeMorProb',
       'nTutorConceptsSession', 'nCorrectKC',
       'nCorrectPSStepSinceLastWrongFirst']

selected_features = PROBLEM_FEATURES

# # # data = data[(data['decisionPoint'] == 'probStart') & (data['critical'] == 1)]
X = train_data[selected_features].values

# output_features = ['normal Bellman Q value for action 0', 
#                    'normal Bellman Q value for action 1', 
#                    'normal Bellman Q value for action 2']
# output_features = ['normal Bellman Q value for action 1']
Y = train_data[output_features].values

# # define feature selection
# fs = SelectKBest(score_func=f_regression, k=30)
# # apply feature selection
# X = fs.fit_transform(X.values, Y)

# data = pd.read_csv('./data/concrete_data.csv', delimiter=',')
# X = data.values[:,:8]
# Y = data.values[:,-1]

# X = load_boston().data[:400]
# Y = load_boston().target[:400]

# from sklearn import preprocessing
# X = preprocessing.normalize(X)
# Y = preprocessing.normalize(Y.reshape(-1, 1).T)
# Y = Y.T

# try:
#     iter(Y[0])
# except TypeError:
#     Y = Y.reshape((Y.shape[0], 1))

def fp_maximals(df):
    # https://towardsdatascience.com/how-to-find-closed-and-maximal-frequent-itemsets-from-fp-growth-861a1ef13e21
    start_time = time.time()
    frequent = fpgrowth(df, min_support=0.2, use_colnames=True)
    print('Time to find frequent itemset')
    print("--- %s seconds ---" % (time.time() - start_time))# Task 2&3: Find closed/max frequent itemset using frequent itemset found in task1
    su = frequent.support.unique()#all unique support count
    #Dictionay storing itemset with same support count key
    fredic = {}
    for i in range(len(su)):
        inset = list(frequent.loc[frequent.support ==su[i]]['itemsets'])
        fredic[su[i]] = inset#Dictionay storing itemset with  support count <= key
    fredic2 = {}
    for i in range(len(su)):
        inset2 = list(frequent.loc[frequent.support<=su[i]]['itemsets'])
        fredic2[su[i]] = inset2#Find Closed frequent itemset
    start_time = time.time()
    
    cl = []
    for index, row in frequent.iterrows():
        isclose = True
        cli = row['itemsets']
        cls = row['support']
        checkset = fredic[cls]
        for i in checkset:
            if (cli!=i):
                if(frozenset.issubset(cli,i)):
                    isclose = False
                    break
        
        if(isclose):
            cl.append(row['itemsets'])
    
    print('Time to find Close frequent itemset')
    print("--- %s seconds ---" % (time.time() - start_time))  
        
    #Find Max frequent itemset
    start_time = time.time()
    ml = []
    
    for index, row in frequent.iterrows():
        isclose = True
        cli = row['itemsets']
        cls = row['support']
        checkset = fredic2[cls]
        for i in checkset:
            if (cli!=i):
                if(frozenset.issubset(cli,i)):
                    isclose = False
                    break
        
        if(isclose):
            ml.append(row['itemsets'])
            
    print('Time to find Max frequent itemset')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return ml

def cluster_membership(X, membership_functions):
    membership_degree = 1.0
    for x, term in zip(X, membership_functions):
        membership_degree = min(membership_degree, term(x))
        # membership_degree *= term(x)
    return membership_degree

def make_cluster_membership_functions(membership_functions):
    # cluster_membership_functions = []
    clusters = []
    for cluster_membership_function in itertools.product(*membership_functions):
        cluster_membership_functions = []
        for membership_function in cluster_membership_function:
            cluster_membership_functions.append(partial(gaussian, center=membership_function['center'], sigma=membership_function['sigma']))
        clusters.append(partial(cluster_membership, membership_functions=cluster_membership_functions))
    return clusters
    # for input_dimension_idx, linguistic_terms in enumerate(membership_functions):
    #     for linguistic_term_idx, linguistic_term in enumerate(linguistic_terms):
            

# the below comments about Pyrenees problem-level decision performance was observed on columns 50 to 130 aka [:,50:130]
# which achieved a RMSE of ~ 0.07023 before fine-tuning and required 441 rules
a = 0.3 # default = 0.2; alpha must be 1e-2 for Pyrenees problem-level decision data for both antecedents and consequents generation
b = 0.5 # default = 0.6; beta must be 0.95 for Pyrenees problem-level decision data for both antecedents and consequents generation
BATCH_SIZE = 200
rules = []
weights = []
antecedents = []
consequents = []
NUM_OF_EPOCHS = 1
NUM_OF_BATCHES = round(len(X) / BATCH_SIZE)
for epoch in range(NUM_OF_EPOCHS):
    replay_buffer_X = deepcopy(X)
    replay_buffer_Y = deepcopy(Y)
    batch_num = 0
    while replay_buffer_X.shape[0] > 0:
        print('Batch %s' % batch_num)
        batch_num += 1
    # for i in range(NUM_OF_BATCHES):
        # print('epoch %s batch %s' % (epoch, i))
        number_of_rows = replay_buffer_X.shape[0]
        random_indices = np.random.choice(number_of_rows, size=BATCH_SIZE, replace=False)
        batch_X = replay_buffer_X[random_indices, :]
        batch_Y = replay_buffer_Y[random_indices, :]
        replay_buffer_X = np.delete(replay_buffer_X, random_indices, axis=0)
        replay_buffer_Y = np.delete(replay_buffer_Y, random_indices, axis=0)
        # batch_X = X[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
        # batch_Y = Y[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
        X_mins = np.min(batch_X, axis=0)
        X_maxes = np.max(batch_X, axis=0)
        Y_mins = np.min(batch_Y, axis=0)
        Y_maxes = np.max(batch_Y, axis=0)
        print('antecedents')
        antecedents = CLIP(batch_X, batch_Y, X_mins, X_maxes, antecedents, alpha=0.01, beta=0.95) # the second argument is currently not used
        print('consequents')
        consequents = CLIP(batch_Y, batch_X, Y_mins, Y_maxes, consequents, alpha=0.01, beta=0.95) # the second argument is currently not used
        # antecedents alpha=0.1 beta=0.5; consequents alpha=0.01 beta=0.95; RMSE of 0.08791; 436 rules; 130 features
        
        # # remove terms not well supported by the data
        # for p in range(X.shape[1]):
        #     supports = [A['support'] for A in antecedents[p]]
        #     indices_of_terms_to_remove = np.where(supports < np.median(supports))[0]
        #     for index in indices_of_terms_to_remove:
        #         antecedents[p][index] = None
        #     idx = 0
        #     terms_removed = 0
        #     while True:
        #         try:
        #             if antecedents[p][idx] is None:
        #                 antecedents[p].pop(idx)
        #                 terms_removed += 1
        #             else:
        #                 idx += 1
        #         except IndexError:
        #             break
    
        # # remove terms not well supported by the data
        # for p in range(Y.shape[1]):
        #     supports = [A['support'] for A in consequents[p]]
        #     indices_of_terms_to_remove = np.where(supports < np.median(supports))[0]
        #     for index in indices_of_terms_to_remove:
        #         consequents[p][index] = None
        #     idx = 0
        #     terms_removed = 0
        #     while True:
        #         try:
        #             if consequents[p][idx] is None:
        #                 consequents[p].pop(idx)
        #                 terms_removed += 1
        #             else:
        #                 idx += 1
        #         except IndexError:
        #             break
        
        # print('terms removed %s' % terms_removed)
        # print('continue?')
        # input()
        # print('continuing.')
        
        # if SHOW_PLOTS:
        #     for p in range(X.shape[1]):
        #         terms = antecedents[p]
        #         for term in terms:
        #             mu = term['center']
        #             sigma = term['sigma']
        #             x = np.linspace(mu - 3*sigma, mu + 3*sigma, 250)
        #             plt.plot(x, gaussian(x, mu, sigma))
        #         plt.title('antecedent %s' % p)
        #         plt.show()
                
        #     for q in range(Y.shape[1]):
        #         terms = consequents[q]
        #         for term in terms:
        #             mu = term['center']
        #             sigma = term['sigma']
        #             x = np.linspace(mu - 3*sigma, mu + 3*sigma, 250)
        #             plt.plot(x, gaussian(x, mu, sigma))
        #         plt.title('consequent %s' % q)
        #         plt.show()
                
        # print('making fuzzy clusters...')
        # # input_space_clusters = make_cluster_membership_functions(antecedents)
        # # print('input space done (%s identified)...' % len(input_space_clusters))
        # output_space_clusters = make_cluster_membership_functions(consequents)
        # print('output space done (%s identified)...' % len(output_space_clusters))
        
        # import random
        # # input_space_clusters = random.sample(input_space_clusters, 250)
        # # output_space_clusters = random.sample(output_space_clusters, 50)
        
        # # # --- INPUT SPACE ---
        
        # # medians = []
        # # for input_cluster in input_space_clusters:
        # #     xs = []
        # #     for x in X:
        # #         xs.append(input_cluster(x))
        # #     medians.append(np.median(xs))
        # #     plt.scatter(range(len(X)), sorted(xs, reverse=True), s=0.75, alpha=0.75)
        # # input_NC = np.mean(medians)
        # # plt.hlines(input_NC, 0, len(X), colors='grey', linestyles='dashed', label='Noise Criterion')
        # # plt.xlabel('Observations')
        # # plt.ylabel('Minimum Degree of Membership to Input Space Clusters')
        # # plt.title('Identification of Scrupulous Data in the Input Space Domain')
        # # plt.legend()
        # # plt.show()
        
        # # --- OUTPUT SPACE ---
        
        # medians = []
        # for output_cluster in output_space_clusters:
        #     ys = []
        #     for y in Y:
        #         ys.append(output_cluster(y))
        #     medians.append(np.median(ys))
        #     plt.scatter(range(len(X)), sorted(ys, reverse=True), s=0.75, alpha=0.75)
        # output_NC = np.mean(medians)
        # plt.hlines(output_NC, 0, len(X), colors='grey', linestyles='dashed', label='Noise Criterion')
        # plt.xlabel('Observations')
        # plt.ylabel('Minimum Degree of Membership to Output Space Clusters')
        # plt.title('Identification of Scrupulous Data in the Output Space Domain')
        # plt.legend()
        # plt.show()
        
        # # NC = (1/(len(input_space_clusters) + len(output_space_clusters)) * min(input_NC, output_NC)
        # # NC = min(input_NC, output_NC)
        # NC = output_NC * 0.007 # this works pretty well, put it back once done with frequent itemset
        # NC = 0
    
        # # remove data according to noise criterion
        # indices_to_remove = []
        # for idx, tupl in enumerate(list(zip(X, Y))):
        #     x = tupl[0]
        #     y = tupl[1]
        #     # noisy_data = noise_criterion(x, y, input_space_clusters, output_space_clusters, NC)
        #     noisy_data = noise_criterion(y, y, output_space_clusters, output_space_clusters, NC)
        #     if noisy_data:
        #         indices_to_remove.append(idx)
                
        # filtered_X = deepcopy(X)
        # filtered_X = np.delete(filtered_X, indices_to_remove, axis=0)
        # filtered_Y = deepcopy(Y)
        # filtered_Y = np.delete(filtered_Y, indices_to_remove, axis=0)    
        # print('size of data now after NC: %s' % len(filtered_X))
        
        # batch_X = filtered_X
        # batch_Y = filtered_Y
        
        filtered_X = batch_X
        filtered_Y = batch_Y
        
        antecedents, consequents, rules, weights = rule_creation(filtered_X, filtered_Y, antecedents, consequents, rules, weights)
            
        print('number of rules %s ' % len(rules))
        consequences = [rules[idx]['C'][0] for idx in range(len(rules))]
        print(np.unique(consequences, return_counts=True))
        
        # make FNN
        all_antecedents_centers = []
        all_antecedents_widths = []
        all_consequents_centers = []
        all_consequents_widths = []
        for p in range(X.shape[1]):
            antecedents_centers = [term['center'] for term in antecedents[p]]
            antecedents_widths = [term['sigma'] for term in antecedents[p]]
            all_antecedents_centers.append(antecedents_centers)
            all_antecedents_widths.append(antecedents_widths)
        for q in range(Y.shape[1]):
            consequents_centers = [term['center'] for term in consequents[q]]
            consequents_widths = [term['sigma'] for term in consequents[q]]
            all_consequents_centers.append(consequents_centers)
            all_consequents_widths.append(consequents_widths)
    
        term_dict = {}
        term_dict['antecedent_centers'] = boolean_indexing(all_antecedents_centers)
        term_dict['antecedent_widths'] = boolean_indexing(all_antecedents_widths)
        term_dict['consequent_centers'] = boolean_indexing(all_consequents_centers)
        term_dict['consequent_widths'] = boolean_indexing(all_consequents_widths)
        
        antecedents_indices_for_each_rule = np.array([rules[k]['A'] for k in range(len(rules))])
        consequents_indices_for_each_rule = np.array([rules[k]['C'] for k in range(len(rules))])
        
        # tmp_data = []
        # for col in range(antecedents_indices_for_each_rule.shape[1]):
        #     tmp_data.append([('%s;%s' % (col, val)) for val in antecedents_indices_for_each_rule[:,col]])
        
        # transformed_antecedents_indices_for_each_rule = np.array(tmp_data).T
        # df = pd.DataFrame(transformed_antecedents_indices_for_each_rule)
        # # one hot encoding
        
        # items = np.unique(transformed_antecedents_indices_for_each_rule)
        # itemset = set(items)
        # encoded_vals = []
        # for index, row in df.iterrows():
        #     rowset = set(row) 
        #     labels = {}
        #     uncommons = list(itemset - rowset)
        #     commons = list(itemset.intersection(rowset))
        #     for uc in uncommons:
        #         labels[uc] = 0
        #     for com in commons:
        #         labels[com] = 1
        #     encoded_vals.append(labels)
        # encoded_vals[0]
        # ohe_df = pd.DataFrame(encoded_vals)
        
        # # from mlxtend.frequent_patterns import apriori
        # # freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
        # # freq_items.head(7)
        
        # # from mlxtend.frequent_patterns import fpgrowth
        # # frequent_itemsets = fpgrowth(ohe_df, min_support=0.6)
        # # from mlxtend.frequent_patterns import association_rules
        # # freq_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        
        # results = fp_maximals(ohe_df)
        
        # tmp_arrays = []
        # for idx, freq_items in enumerate(results):
        #     freq_items_lst = list(freq_items)
        #     tmp_array = np.empty(transformed_antecedents_indices_for_each_rule.shape[1])
        #     tmp_array[:] = np.nan
        #     for inner_idx, freq_item in enumerate(freq_items):
        #         freq_item_tuple = tuple(int(val) for val in freq_item.split(';'))
        #         col_idx = freq_item_tuple[0]
        #         term_idx = freq_item_tuple[1]
        #         tmp_array[col_idx] = term_idx
        #     tmp_arrays.append(tmp_array)
            
        
        print('FNN built.')
        
        from safin import SaFIN
        
        fnn = SaFIN(term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule)
        
        print('making predictions...')
        
        y_predicted = fnn.feedforward(batch_X)
            
        # from safin import SaFIN as old_SaFIN
        
        # old_fnn = old_SaFIN(term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule)
    
        # old_y_predicted = []
        # for tupl in zip(batch_X, batch_Y):
        #     x = tupl[0]
        #     d = tupl[1]
        #     old_y_predicted.append((old_fnn.feedforward(x)))
    
            
        from sklearn.metrics import mean_squared_error
        # rmse = (np.sqrt(mean_squared_error(Y[:,0].tolist(), y_predicted[:,0].tolist())))
        init_rmse = (np.sqrt(mean_squared_error(batch_Y, y_predicted)))
        print('rmse before tuning %s' % init_rmse)
        
        # delete low weight rules
        
        if replay_buffer_X.shape[0] != 0:
            import random
            choices = list(np.where(np.array(weights) <= np.median(weights))[0])
            if len(choices) < BATCH_SIZE:
                NUM_TO_DELETE = len(choices)
            # elif len(choices) > BATCH_SIZE:
            #     NUM_TO_DELETE = len(choices)
            else:
                NUM_TO_DELETE = BATCH_SIZE
            rule_indices = random.sample(choices, k=NUM_TO_DELETE)
            
            rules = [rule for i, rule in enumerate(rules) if i not in rule_indices]
            weights = [weight for i, weight in enumerate(weights) if i not in rule_indices]
        
        # # print('hit "enter" to continue...')
        # # input()
        # # print('continuing.')
        
        # l_rate = 0.1 # 0.1 was used for Pyrenees data
        # n_epoch = 1000
        # epsilon = 0.25
        # epoch = 0
        # curr_rmse = init_rmse
        # prev_rmse = init_rmse
        # while curr_rmse <= prev_rmse:
        #     # print('epoch %s' % epoch)
        #     y_predicted = []
        #     deltas = None
        #     for idx, x in enumerate(batch_X):
        #         # print(epoch, idx)
        #         y = batch_Y[idx][0]
        #         # y = Y[idx]
                
        #         # if idx == 59:
        #         #     print('wait')
        #         iterations = 1
        #         while True:
        #             o5 = fnn.feedforward(x)
        #             consequent_delta_c, consequent_delta_widths, antecedent_delta_c, antecedent_delta_widths = fnn.backpropagation(x, y)
        #             if deltas is None:
        #                 deltas = {'c_c':consequent_delta_c, 'c_w':consequent_delta_widths, 'a_c':antecedent_delta_c, 'a_w':antecedent_delta_widths}
        #             else:
        #                 deltas['c_c'] += consequent_delta_c
        #                 deltas['c_w'] += consequent_delta_widths
        #                 deltas['a_c'] += antecedent_delta_c
        #                 deltas['a_w'] += antecedent_delta_widths
        #             break
        #             # if np.abs(o5 - y) < epsilon or iterations >= 250:
        #             #     y_predicted.append(o5)
        #             #     print('achieved with %s and %s iterations' % (np.abs(o5 - y), iterations))
        #             #     break
        #             # else:
        #             #     # print(np.abs(o5 - y))
        #             #     consequent_delta_c, consequent_delta_widths, antecedent_delta_c, antecedent_delta_widths = fnn.backpropagation(x, y)
        #             #     # print(consequent_delta_c)
        #             #     # print(consequent_delta_widths)
        #             #     # print(antecedent_delta_c)
        #             #     # print(antecedent_delta_widths)
                        
        #             #     # fnn.term_dict['consequent_centers'] += 1e-4 * consequent_delta_c
        #             #     # fnn.term_dict['consequent_widths'] += 1e-8 * consequent_delta_widths
        #             #     # fnn.term_dict['antecedent_centers'] += 1e-4 * antecedent_delta_c
        #             #     # fnn.term_dict['antecedent_widths'] += 1e-8 * antecedent_delta_widths
                        
        #             #     fnn.term_dict['consequent_centers'] += l_rate * consequent_delta_c
        #             #     # fnn.term_dict['consequent_widths'] += 1.0 * l_rate * consequent_delta_widths
        #             #     # fnn.term_dict['antecedent_centers'] += l_rate * antecedent_delta_c
        #             #     # fnn.term_dict['antecedent_widths'] += 1.0 * l_rate * antecedent_delta_widths
                        
        #             #     # remove anything less than or equal to zero for the linguistic term widths
        #             #     # if (fnn.term_dict['consequent_widths'] <= 0).any() or (fnn.term_dict['antecedent_widths'] <= 0).any():
        #             #     #     print('fix weights')
        #             #     # fnn.term_dict['consequent_widths'][fnn.term_dict['consequent_widths'] <= 0.0] = 1e-1
        #             #     # fnn.term_dict['antecedent_widths'][fnn.term_dict['antecedent_widths'] <= 0.0] = 1e-1
                        
        #             #     iterations += 1
        #     fnn.term_dict['consequent_centers'] -= l_rate * (deltas['c_c'] / len(batch_X))
        #     fnn.term_dict['consequent_widths'] -= l_rate * (deltas['c_w'] / len(batch_X))
        #     fnn.term_dict['antecedent_centers'] -= l_rate * (deltas['a_c'] / len(batch_X))
        #     fnn.term_dict['antecedent_widths'] -= l_rate * (deltas['a_w'] / len(batch_X))
            
        #     y_predicted = []
        #     for tupl in zip(batch_X, batch_Y):
        #         x = tupl[0]
        #         d = tupl[1]
        #         y_predicted.append((fnn.feedforward(x)))
            
        #     prev_rmse = curr_rmse
        #     curr_rmse = (np.sqrt(mean_squared_error(batch_Y[:,0].tolist(), y_predicted)))
        #     print('--- epoch %s --- rmse after tuning = %s (prev rmse was %s; init rmse was %s)' % (epoch, curr_rmse, prev_rmse, init_rmse))
        #     epoch += 1
        
        #     if curr_rmse > prev_rmse:
        #         # reverse the updates
        #         fnn.term_dict['consequent_centers'] += l_rate * (deltas['c_c'] / len(batch_X))
        #         fnn.term_dict['consequent_widths'] += l_rate * (deltas['c_w'] / len(batch_X))
        #         fnn.term_dict['antecedent_centers'] += l_rate * (deltas['a_c'] / len(batch_X))
        #         fnn.term_dict['antecedent_widths'] += l_rate * (deltas['a_w'] / len(batch_X))
    
        # print('hit "enter" to continue...')
        # input()
        # print('continuing.')
    
test_X = test_data[selected_features].values
# Y = data['normal Bellman Q value for action 0'].values
test_Y = test_data[output_features].values
        
# test_X = load_boston().data[400:]
# test_Y = load_boston().target[400:]
import time
start = time.time()
y_predicted = fnn.feedforward(test_X[0:1000])
end= time.time()
print('fnn test rmse %s' % (np.sqrt(mean_squared_error(test_Y[0:1000], y_predicted))))

# knn = KNeighborsRegressor(n_neighbors=5)
# knn.fit(X, Y)
# knn_rmse = (np.sqrt(mean_squared_error(Y[:,0].tolist(), knn.predict(X))))
# print('knn train rmse %s' % knn_rmse)

# knn_rmse = (np.sqrt(mean_squared_error(test_Y.tolist(), knn.predict(test_X))))
# print('knn test rmse %s' % knn_rmse)