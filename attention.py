# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import tensorflow as tf 

from tensorflow.math import exp, multiply, sqrt, maximum
from tensorflow import Variable
from tensorflow import Parameter

from myutils import dynamic_softmax

class Attention(tf.keras.layers):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, device=tf.device("/cpu:0")):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(Attention, self).__init__()
        self.attention_size = attention_size
        self.device = device
        self.attention_vector = Variable(attention_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.attention_size)
        for weight in self.weights:
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({attention_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        logits = inputs.multiply(self.attention_vector) # all_utter * max_sen_len * dim_sen dim_sen=all_utter * max_sen_len
        unnorm_ai = tf.exp(logits - logits.maximum()) # all_utter * max_sen_len

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)
        idxes = tf.expand_dims(tf.range(0, max_len, out=Tensor(max_len)), 0)
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float())
        mask = mask.to(self.device) if self.device else mask

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.reduce_sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums) # scalar utter*max_sen_len

        # apply attention weights
        weighted = multiply(inputs, tf.expand_dims(attentions, -1).tf.broadcast_to(inputs)) # utter*max_sen_len*dim_sen utter*max_sen_len*dim_sen=utter*max_sen_len*dim_sen

        # get the final fixed vector representations of the sentences
        representations = tf.reduce_sum(weighted, 1) # utter*dim_sen

        return representations, attentions
