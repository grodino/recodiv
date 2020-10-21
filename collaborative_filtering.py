import numpy as np
import tensorflow as tf

user_input = tf.keras.layers.Input(shape=[1], dtype=tf.float32)
item_input = tf.keras.layers.Input(shape=[1], dtype=tf.float32)

# Number of latent features
mf_dim = 10 
# Facteur de régularisation l2
mf_regularization = 0.1

# Initial weight values
embedding_initializer = "glorot_uniform"

embedding_user = tf.keras.layers.Embedding(
    len(graph.ids[0]) + 1,
    mf_dim,
    embeddings_initializer=embedding_initializer,
    embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
    name="embedding_user"
)(user_input)

embedding_item = tf.keras.layers.Embedding(
    len(graph.ids[1]) + 1,
    mf_dim,
    embeddings_initializer=embedding_initializer,
    embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
    name="embedding_item"
)(item_input)

rating_prediction = tf.keras.layers.Dot(name="Dot-Product", axes=1)([
    tf.keras.layers.Flatten()(embedding_user), 
    tf.keras.layers.Flatten()(embedding_item)
])
model = tf.keras.models.Model([user_input, item_input], rating_prediction)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=not tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanSquaredError()]
)
# print(model.output_dim)

model.fit(
    x=[users, listened_songs],
    y=listenings,
    verbose=True
)