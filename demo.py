import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import json
from utils import *
from bpr import *
import tensorflow as tf
import hashlib


user_hashes = 'dbfe937585b4147cb6f305277685021c22cafe00bdc241fd870c500d3e87cb2b'
pass_hashes = '98a4fec3f9b6a6d43df99b176a32b32a8a668b12cff6d20809e55ee26cae2c34'

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

def get_recommend(model, user_id, k=10):
    user_vecs = model.get_layer('user_embedding').get_weights()[0]
    item_vecs = model.get_layer('item_embedding').get_weights()[0]
    
    transaction = user_history_df['sid'][user_id]
    user_vector = user_vecs[user_id]
    rec = (np.dot(user_vector,
                    item_vecs.T))
    rec[transaction] = -np.inf

    top_k = np.argsort(-rec)[:k]
    return top_k, rec[top_k]

st.title("Welcome to Galaxy Play 🍔📸")
st.header("Suggest movies for users!")

user_history_df = pd.read_pickle('data.pkl')




def main():
    user_id = st.sidebar.selectbox(
        'Which user id would you recommend?',
        user_history_df.index
    )

    k = st.sidebar.slider('Top k items',1, 20, value=10)

    def get_history(user_id):
        movies_df = pd.read_pickle('movies.pkl')
        sid = user_history_df['sid'][user_id]
        return movies_df.loc[sid, ['movie_name', 'movie_genres']]

    if st.checkbox('Show user history'):
        df = get_history(user_id)
        st.write('History of user have id: ', user_id, f'(include {df.shape[0]} items )')
        st.dataframe(data=df, width=2000, height=480)
        


    b_rules = st.checkbox('Recommend using Association Rules')
    if b_rules:
        min_conf = st.slider('Association Rules Min Confidence',0., 1., value=0.1, step=0.01)
        rules = pd.read_pickle('rules.pkl')
        rule = ruleSelection(rules, min_confidence=min_conf)
        if rule.shape[0]:
            movies_df = pd.read_pickle('movies.pkl')
            result = recommend_items(user_history_df['sid'][user_id], rule, movies=movies_df, verbose=True, topk=k)
            
            st.dataframe(result)
        else:
            st.write(f'No Suggestion with confidence: {min_conf}')
        if st.button('Show Rules'):
            display_rules = rule
            display_rules.antecedents = display_rules.antecedents.astype(str)
            display_rules.consequents = display_rules.consequents.astype(str)        
            st.dataframe(data=display_rules, width=2000, height=480)


    b_bpr = st.checkbox('Recommend using BPR MF')
    if b_bpr:
        f = open('bpr/bpr_model.json')
        config = json.load(f)
        model = build_model(config['M'], config['N'], config['k'])
        model.load_weights("bpr/bpr_model.h5")
        sids, scores = get_recommend(model, user_id, k)

        movies_df = pd.read_pickle('movies.pkl')
        res = movies_df.loc[sids, ['movie_name', 'movie_genres']].copy()
        res['score'] = scores
        st.dataframe(res)

    k_from_emb = st.sidebar.slider('Top k from Embedding vector',1, 10, value=3)


    b_item_emb = st.checkbox('Recommend using Item embedding')
    if b_item_emb:
        f = open('bpr/bpr_model.json')
        config = json.load(f)
        model = build_model(config['M'], config['N'], config['k'])
        model.load_weights("bpr/bpr_model.h5")
        item_id = user_history_df['sid'][user_id]
        time_duration = user_history_df['time_by_duration'][user_id]
        sids,_ = get_similar_items(model, item_id, time_duration, topk=k_from_emb)
        movies_df = pd.read_pickle('movies.pkl')
        res = movies_df.loc[sids, ['movie_name', 'movie_genres']].copy()
        st.dataframe(res)


if __name__ == '__main__':
    text_input_container1 = st.empty()
    text_input_container2 = st.empty()

    username = text_input_container1.text_input("User Name")
    password = text_input_container2.text_input("Password",type='password')
  
    if check_hashes(username, user_hashes) & check_hashes(password, pass_hashes):
        text_input_container1.empty()
        text_input_container2.empty()
        main()