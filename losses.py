import tensorflow as tf

# Sum of binary cross entropy used for literal channel, implied channel, and overall sarcasm analyzer.
def bce(y_true, y_pred):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss = loss_fn(y_true, y_pred)
    return tf.reduce_sum(tf.cast(loss, tf.float32))

# Overall model loss is weighted average of components' losses, with weights chosen as a hyperparameter.
def total_loss(lit_true, lit_pred, imp_true, imp_pred, sarc_true, sarc_pred, loss_coefs = [1/3, 1/3, 1/3]):
    lit_loss = bce(lit_true, lit_pred)
    imp_loss = bce(imp_true, imp_pred)
    sarc_loss = bce(sarc_true, sarc_pred)
    return loss_coefs[0] * lit_loss + loss_coefs[1] * imp_loss + loss_coefs[2] * sarc_loss 
    


