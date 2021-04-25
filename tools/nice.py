'''
Non-Linear Independent Component Estimination (NICE) code

Modified from
- https://github.com/bojone/flow

2019-09-02 first created
2020-04-15 updated
2020-04-28 modifed from https://github.com/jaekookang/flow_based_models/blob/master/models/nice.py
'''
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
K = tfk.backend


class AdditiveAffineLayer(tfkl.Layer):
    def __init__(self, inp_dim, shuffle_type, n_couple_layer, n_hid_layer, n_hid_dim, name):
        super(AdditiveAffineLayer, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.shuffle_type = shuffle_type
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        # self.g = self._g(add_batchnorm=False)
        self.g = self._conv(add_batchnorm=False)
        # self.s = self._g(add_batchnorm=False)
        # self.t = self._g(add_batchnorm=False)
        self.idx = tf.Variable(list(range(self.inp_dim)),
                               shape=(self.inp_dim,),
                               trainable=False,
                               name='index',
                               dtype='int64')
        if self.shuffle_type == 'random':
            self.idx.assign(tf.random.shuffle(self.idx))
        elif self.shuffle_type == 'reverse':
            self.idx.assign(tf.reverse(self.idx, axis=[0]))
        else:
            raise NotImplementedError

    def call(self, x):
        # Forward
        x = self.shuffle(x)
        x1, x2 = self.split(x)
        mx1 = self.g(x1)
        # s = self.s(x1)
        # t = self.t(x1)
        # x1, x2 = self.couple([x1, x2, s, t])
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])
        return x

    def shuffle(self, x, isInverse=False):
        if not isInverse:
            # Forward
            idx = self.idx
        else:
            # Inverse
            idx = tf.map_fn(tf.math.invert_permutation,
                            tf.expand_dims(self.idx, 0))
            idx = tf.squeeze(idx)
        x = tf.transpose(x)
        x = tf.gather(x, idx)
        x = tf.transpose(x)
        return x

    def split(self, x):
        dim = x.shape[-1]
        x = tf.reshape(x, [-1, dim//2, 2])
        return [x[:, :, 0], x[:, :, 1]]  # (N,dim//2) (N,dim//2)

    def _g(self, add_batchnorm=False, name=None):
        mlp = tfk.Sequential(name=name)
        for _ in range(self.n_hid_layer):
            mlp.add(tfkl.Dense(self.n_hid_dim, activation=tfkl.LeakyReLU()))
            if add_batchnorm:
                mlp.add(tfkl.BatchNormalization())
        mlp.add(tfkl.Dense(self.inp_dim//2, activation='linear'))
        return mlp
    
    def _conv(self, add_batchnorm=False, name=None):
        conv = tfk.Sequential(name=name)
        conv.add(tfkl.Lambda(lambda x: tf.expand_dims(x, axis=-1)))
        for _ in range(self.n_hid_layer):
            conv.add(tfkl.Conv1D(self.inp_dim*2, 3, padding='same'))
        conv.add(tfkl.LeakyReLU(alpha=0.02))
        conv.add(tfkl.AveragePooling1D())
        conv.add(tfkl.Flatten())
        if add_batchnorm:
            conv.add(tfkl.BatchNormalization())
        conv.add(tfkl.Dense(self.inp_dim//2, activation='linear'))
        return conv

    def couple(self, xs, isInverse=False):
        x1, x2, mx1 = xs
        # x1, x2, s, t = xs
        if isInverse:
            return [x1, x2-mx1] # Inverse
            # return [x1, (x2 - t) * tf.math.exp(-s)]
        else:
            return [x1, x2+mx1] # Forward
            # return [x1, x2 * tf.math.exp(s) + t]


    def concat(self, xs):
        xs = [tf.expand_dims(x, 2) for x in xs]  # [(N,392) (N,392)]
        x = tf.concat(xs, 2)  # (N,dim,2)
        return tf.reshape(x, [-1, tf.math.reduce_prod(x.shape[1:])])

    def inverse(self, x):
        x1, x2 = self.split(x)
        mx1 = self.g(x1)
        # s = self.s(x1)
        # t = self.t(x1)
        # x1, x2 = self.couple([x1, x2, s, t], isInverse=True)
        x1, x2 = self.couple([x1, x2, mx1], isInverse=True)
        x = self.concat([x1, x2])
        x = self.shuffle(x, isInverse=True)
        return x


class Scale(tfkl.Layer):
    def __init__(self, inp_dim, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.inp_dim = inp_dim
        self.scaling = self.add_weight(name='scaling',
                                       shape=(1, self.inp_dim),
                                       initializer='glorot_normal',
                                       trainable=True)

    def call(self, x):
        self.add_loss(-tf.math.reduce_sum(self.scaling))
        return tf.math.exp(self.scaling) * x

    def inverse(self, x):
        scale = tf.math.exp(-self.scaling)
        return scale * x


class Sigmoid(tfkl.Layer):
    def __init__(self, inp_dim, ndim_z, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.inp_dim = inp_dim
        self.ndim_z = ndim_z

    def call(self, x):
        # x is assumed to be [z, pad, y] where z is at the end
        z, x = x[:, :self.ndim_z], x[:, self.ndim_z:]  # split
        x_sigmoid = tfk.activations.sigmoid(x)
        x = tf.concat([z, x_sigmoid], axis=-1) # combine
        return x

    def inverse(self, x):
        # x is assumed to be [z, pad, y] where z is at the end
        
        #x_sigmoid = tf.clip_by_value(x_sigmoid, K.epsilon(), 1. - K.epsilon())
        #x = tf.math.log(tf.math.divide(x_sigmoid, (1. - x_sigmoid)))

        z, x_sigmoid = x[:, :self.ndim_z], x[:, self.ndim_z:] # split
        x_sigmoid = tf.clip_by_value(x_sigmoid, K.epsilon(), 1. - K.epsilon())
        x = tf.math.log(tf.math.divide(x_sigmoid, (1. - x_sigmoid)))
        x = tf.concat([z, x], axis=-1) # combine
        return x


class NICECouplingBlock(tfk.Model):
    def __init__(self, inp_dim, shuffle_type, n_couple_layer, n_hid_layer, n_hid_dim, name='NICECouplingBlock'):
        super(NICECouplingBlock, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.shuffle_type = shuffle_type
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.AffineLayers = []
        for i in range(self.n_couple_layer):
            layer = AdditiveAffineLayer(self.inp_dim,
                                        self.shuffle_type,
                                        self.n_couple_layer,
                                        self.n_hid_layer,
                                        self.n_hid_dim,
                                        name=f'layer{i}')
            self.AffineLayers += [layer]

        self.scale = Scale(self.inp_dim, name='ScaleLayer')
        self.AffineLayers += [self.scale]

    def call(self, x):
        act = x
        for i in range(self.n_couple_layer):
            act = self.AffineLayers[i](act)
        act = self.scale(act)
        return act

    def inverse(self, y):
        act = y
        act = self.scale.inverse(act)
        for i in reversed(range(self.n_couple_layer)):
            act = self.AffineLayers[i].inverse(act)
        return act


class NICECouplingBlockSigmoid(tfk.Model):
    def __init__(self, inp_dim, shuffle_type, n_couple_layer, n_hid_layer, n_hid_dim, z_dim=2, name='NICECouplingBlockSigmoid'):
        super(NICECouplingBlockSigmoid, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.shuffle_type = shuffle_type
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.AffineLayers = []
        for i in range(self.n_couple_layer):
            layer = AdditiveAffineLayer(self.inp_dim,
                                        self.shuffle_type,
                                        self.n_couple_layer,
                                        self.n_hid_layer,
                                        self.n_hid_dim,
                                        name=f'layer{i}')
            self.AffineLayers += [layer]

        self.scale = Scale(self.inp_dim, name='ScaleLayer')
        self.sigmoid = Sigmoid(self.inp_dim, self.z_dim, name='SigmoidLayer')
        self.AffineLayers += [self.scale]
        self.AffineLayers += [self.sigmoid]

    def call(self, x):
        act = x
        for i in range(self.n_couple_layer):
            act = self.AffineLayers[i](act)
        act = self.scale(act)
        act = self.sigmoid(act)
        return act

    def inverse(self, y):
        act = y
        act = self.sigmoid.inverse(act)
        act = self.scale.inverse(act)
        for i in reversed(range(self.n_couple_layer)):
            act = self.AffineLayers[i].inverse(act)
        return act        

    
def nice_loss(y_true, y_pred):
    '''Loss function for NICE model'''
    return tf.math.reduce_sum(0.5*y_pred**2)