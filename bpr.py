from tensorflow import keras
from tensorflow.keras.layers import * 
from tensorflow.keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.regularizers import l2

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def bpr_triplet_loss(X):
    positive_item_latent, negative_item_latent, user_latent = X
    xui = K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True)
    xuj = K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True)
    xuij = xui - xuj
    return -tf.reduce_mean(K.log(K.sigmoid(xuij)))

def build_model(num_users, num_items, latent_dim):
    
    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')
    user_input = Input((1, ), name='user_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(num_items, latent_dim, name='item_embedding', input_length=1, embeddings_regularizer=l2(1e-7))
    
    positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))

    user_embedding = Flatten()(Embedding(num_users, latent_dim, name='user_embedding', input_length=1, embeddings_regularizer=l2(1e-7))(user_input))

    loss = keras.layers.Lambda(bpr_triplet_loss, name="lambda_layer")([positive_item_embedding, negative_item_embedding, user_embedding])

    model = Model([positive_item_input, negative_item_input, user_input], loss, name="bprmf")
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss=identity_loss, optimizer=opt)

    return model
