#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:39:43 2021

@author: john
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hyfis import rule_creation
from self_organizing import CLIP
from common import gaussian, boolean_indexing

from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor

# data = pd.read_csv('./data/concrete_data.csv', delimiter=',')
# X = data.values[:,:8]
# Y = data.values[:,-1]

X = load_boston().data[:400]
Y = load_boston().target[:400]

# from sklearn import preprocessing
# X = preprocessing.normalize(X)
# Y = preprocessing.normalize(Y.reshape(-1, 1).T)
# Y = Y.T

try:
    iter(Y[0])
except TypeError:
    Y = Y.reshape((Y.shape[0], 1))

a = 1e-1
b = 8e-1
batch_size = len(X)
rules = []
weights = []
antecedents = []
consequents = []
for i in range(1):
    batch_X = X[batch_size*i:batch_size*(i+1)]
    batch_Y = Y[batch_size*i:batch_size*(i+1)]
    X_mins = np.min(batch_X, axis=0)
    X_maxes = np.max(batch_X, axis=0)
    Y_mins = np.min(batch_Y, axis=0)
    Y_maxes = np.max(batch_Y, axis=0)
    antecedents = CLIP(batch_X, batch_Y, X_mins, X_maxes, antecedents, alpha=a, beta=b) # the second argument is currently not used
    consequents = CLIP(batch_Y, batch_X, Y_mins, Y_maxes, consequents, alpha=a, beta=b) # the second argument is currently not used
    
    for p in range(X.shape[1]):
        terms = antecedents[p]
        for term in terms:
            mu = term['center']
            sigma = term['sigma']
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 250)
            plt.plot(x, gaussian(x, mu, sigma))
        plt.title('antecedent %s' % p)
        plt.show()
        
    for q in range(Y.shape[1]):
        terms = consequents[q]
        for term in terms:
            mu = term['center']
            sigma = term['sigma']
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 250)
            plt.plot(x, gaussian(x, mu, sigma))
        plt.title('consequent %s' % q)
        plt.show()
        
        rules, weights = rule_creation(batch_X, batch_Y, antecedents, consequents, rules, weights)
        
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
        consequents_indices_for_each_rule = np.array([rules[k]['C'] for k in range(len(rules))]).reshape(-1)
        
        from safin import SaFIN
        
        fnn = SaFIN(term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule)

        y_predicted = []
        for tupl in zip(batch_X, batch_Y):
            x = tupl[0]
            d = tupl[1]
            y_predicted.append((fnn.feedforward(x)))
            
        from sklearn.metrics import mean_squared_error
        # rmse = (np.sqrt(mean_squared_error(Y[:,0].tolist(), y_predicted[:,0].tolist())))
        init_rmse = (np.sqrt(mean_squared_error(batch_Y[:,0].tolist(), y_predicted)))
        print('rmse before tuning %s' % init_rmse)
        
        input()
        
        l_rate = 0.1
        n_epoch = 100
        epsilon = 0.25
        epoch = 0
        curr_rmse = init_rmse
        prev_rmse = init_rmse
        while curr_rmse <= prev_rmse:
            # print('epoch %s' % epoch)
            y_predicted = []
            deltas = None
            for idx, x in enumerate(batch_X):
                # print(epoch, idx)
                y = batch_Y[idx][0]
                # y = Y[idx]
                
                # if idx == 59:
                #     print('wait')
                iterations = 1
                while True:
                    o5 = fnn.feedforward(x)
                    consequent_delta_c, consequent_delta_widths, antecedent_delta_c, antecedent_delta_widths = fnn.backpropagation(x, y)
                    if deltas is None:
                        deltas = {'c_c':consequent_delta_c, 'c_w':consequent_delta_widths, 'a_c':antecedent_delta_c, 'a_w':antecedent_delta_widths}
                    else:
                        deltas['c_c'] += consequent_delta_c
                        deltas['c_w'] += consequent_delta_widths
                        deltas['a_c'] += antecedent_delta_c
                        deltas['a_w'] += antecedent_delta_widths
                    break
                    # if np.abs(o5 - y) < epsilon or iterations >= 250:
                    #     y_predicted.append(o5)
                    #     print('achieved with %s and %s iterations' % (np.abs(o5 - y), iterations))
                    #     break
                    # else:
                    #     # print(np.abs(o5 - y))
                    #     consequent_delta_c, consequent_delta_widths, antecedent_delta_c, antecedent_delta_widths = fnn.backpropagation(x, y)
                    #     # print(consequent_delta_c)
                    #     # print(consequent_delta_widths)
                    #     # print(antecedent_delta_c)
                    #     # print(antecedent_delta_widths)
                        
                    #     # fnn.term_dict['consequent_centers'] += 1e-4 * consequent_delta_c
                    #     # fnn.term_dict['consequent_widths'] += 1e-8 * consequent_delta_widths
                    #     # fnn.term_dict['antecedent_centers'] += 1e-4 * antecedent_delta_c
                    #     # fnn.term_dict['antecedent_widths'] += 1e-8 * antecedent_delta_widths
                        
                    #     fnn.term_dict['consequent_centers'] += l_rate * consequent_delta_c
                    #     # fnn.term_dict['consequent_widths'] += 1.0 * l_rate * consequent_delta_widths
                    #     # fnn.term_dict['antecedent_centers'] += l_rate * antecedent_delta_c
                    #     # fnn.term_dict['antecedent_widths'] += 1.0 * l_rate * antecedent_delta_widths
                        
                    #     # remove anything less than or equal to zero for the linguistic term widths
                    #     # if (fnn.term_dict['consequent_widths'] <= 0).any() or (fnn.term_dict['antecedent_widths'] <= 0).any():
                    #     #     print('fix weights')
                    #     # fnn.term_dict['consequent_widths'][fnn.term_dict['consequent_widths'] <= 0.0] = 1e-1
                    #     # fnn.term_dict['antecedent_widths'][fnn.term_dict['antecedent_widths'] <= 0.0] = 1e-1
                        
                    #     iterations += 1
            fnn.term_dict['consequent_centers'] += l_rate * (deltas['c_c'] / len(batch_X))
            fnn.term_dict['consequent_widths'] += l_rate * (deltas['c_w'] / len(batch_X))
            fnn.term_dict['antecedent_centers'] += 1e-8 * (deltas['a_c'] / len(batch_X))
            fnn.term_dict['antecedent_widths'] += 1e-8 * (deltas['a_w'] / len(batch_X))
            
            y_predicted = []
            for tupl in zip(batch_X, batch_Y):
                x = tupl[0]
                d = tupl[1]
                y_predicted.append((fnn.feedforward(x)))
            
            prev_rmse = curr_rmse
            curr_rmse = (np.sqrt(mean_squared_error(batch_Y[:,0].tolist(), y_predicted)))
            print('--- epoch %s --- rmse after tuning = %s (prev rmse was %s; init rmse was %s)' % (epoch, curr_rmse, prev_rmse, init_rmse))
            epoch += 1
        
        

        print('hit "enter" to continue...')
        input()
        
test_X = load_boston().data[400:]
test_Y = load_boston().target[400:]
        
y_predicted = []
for tupl in zip(test_X, test_Y):
    x = tupl[0]
    d = tupl[1]
    y_predicted.append((fnn.feedforward(x)))

print('test rmse %s' % (np.sqrt(mean_squared_error(test_Y, y_predicted))))

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X, Y)
knn_rmse = (np.sqrt(mean_squared_error(Y[:,0].tolist(), knn.predict(X))))
print('knn rmse %s' % knn_rmse)