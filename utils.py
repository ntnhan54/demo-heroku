# import mlxtend
import pandas as pd
import os
import numpy as np
# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# from tqdm import tqdm_notebook
# from itertools import combinations
import math
from scipy.special import softmax
# tqdm.pandas()

def isSublist(a, b):
    return set(a) <= set(b)

def getBasket(transaction, range=(2,10)):
    basket_size = np.random.randint(*range, size = 100)

    n_transaction = len(transaction)
    total = 0
    basket_list = []
    for i in basket_size:
        basket_list += [transaction[total: total + i]]
        total += i
        if total >= n_transaction:
            break
    if len(basket_list[-1]) < range[0]:
        for basket in basket_list:
            if len(basket) < range[1]:
                basket += basket_list.pop()
                break
    return basket_list

def singlelist(x):
    a = []
    order = []
    for idx, sub in enumerate(x):
        if type(sub) is list:
            for i in sub:
                if i not in a:
                    a += [int(i)]
                    order += [idx]
        elif sub not in a:
            a += [int(sub)]
            order += [idx]
    return a, order

# def getBasket(transaction, range=(2,10)):
#     basket_size = np.random.randint(*range, size = 100)

#     n_transaction = len(transaction)
#     total = 0
#     basket_list = []
#     for i in basket_size:
#         basket_list += [transaction[total: total + i]]
#         total += i
#         if total >= n_transaction:
#         break
#     if len(basket_list[-1]) < range[0]:
#         for basket in basket_list:
#             if len(basket) < 10:
#                 basket += basket_list.pop()
#                 break
#     return basket_list


def ruleSelection(rules, antecedent_len=-1, consequents_len=-1, min_confidence=-1, min_lift=-1):

    res = rules.copy()
    if antecedent_len > 0:  
        res['antecedent_len'] = res.antecedents.apply(lambda x: len(x))
        res = res[res.antecedent_len <= antecedent_len]
        res.drop(columns = ['antecedent_len'], inplace = True)
    if consequents_len > 0:
        res['consequents_len'] = res.consequents.apply(lambda x: len(x))
        res = res[res.consequents_len <= consequents_len]
        res.drop(columns = ['consequents_len'], inplace = True)
    if min_confidence > 0:
        res = res[res.confidence >= min_confidence]

    if min_lift > 0:
        res = res[res.lift > min_lift]

    res.sort_values(['confidence', 'lift'], ascending=False, inplace=True)
    res.reset_index(drop=True, inplace=True)

    return res

# rules = pd.read_csv('rules.csv')

# rules.antecedents = rules.antecedents.apply(ast.literal_eval)
# rules.consequents	= rules.consequents.apply(ast.literal_eval)

def recommend_items(user_trans, rule,  movies=None, verbose=False, topk=10):
    
    condition = rule['antecedents'].apply(lambda x: isSublist(x, user_trans))
    pred = rule[condition]
    
    recommend = pred.consequents.tolist()
    recommend, idx = singlelist(recommend)
    conf = pred.confidence.tolist()
    conf = [conf[i] for i in idx]
    
    rcm = []
    cnf = []
    for i in range(len(recommend)):
        if recommend[i] not in user_trans:
            rcm += [recommend[i]]
            cnf += [conf[i]]        


    if not verbose:
        return rcm
    
    rcm = rcm[:topk]
    cnf = cnf[:topk]
        # watched_movies = movies[movies.sid.isin(user_trans)].title.values
    # print('Suggestion for user who watched: ', watched_movies)
    res = []
    for item in rcm:
        r = movies.iloc[[item]]
        res.append(r)
    if len(res) == 0:
        return None
    res = pd.concat(res).reset_index(drop=True)
    res['confidence'] = cnf
    return res[['movie_id','movie_name', 'movie_genres', 'confidence']]

def get_similar_items(model, item_id, time_duration, topk=5):
    item_vecs = model.get_layer('item_embedding').get_weights()[0]
    item_vec = item_vecs[item_id]              
    w = np.array(time_duration)
    score = item_vec.dot(item_vecs.T)

    score[:, item_id] = -np.inf
    y_pred = np.argsort(-score)
    y_pred = y_pred[:, :topk]
    y_pred = y_pred.reshape(-1)
    score = score[:, :topk]
    rank_values = np.array(list(range(topk, 0, -1)) * w.shape[0]).reshape((w.shape[0], -1))
    rank_values = rank_values * softmax(w.reshape((-1,1)))
    # rank_values = rank_values * w.reshape((-1,1))
    rank_values = rank_values.reshape(-1)
    rank_values = np.argsort(-rank_values)
    print(rank_values)
    return singlelist(y_pred[rank_values])