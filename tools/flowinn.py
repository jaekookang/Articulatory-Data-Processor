'''
Flow based models
2020-11-18 first created
'''

import tensorflow as tf
from flowinn_utils import *

tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
K = tfk.backend


def gen(x_data, y_data, xdim, ydim, zdim, dim_tot):
    # X = [[xdim, 0]]
    # Y = [[ydim, 0, zdim]]
    N = x_data.shape[0]
    xpad = dim_tot - xdim
    ypad = dim_tot - ydim - zdim
    X = np.zeros((N, dim_tot)).astype('float32')
    Y = np.zeros((N, dim_tot)).astype('float32')

    X[:, :xdim] = x_data
    Y[:, :ydim] = y_data
    Y[:, ydim+ypad:ydim+ypad+zdim] = np.random.randn(N, zdim)

    for x, y in zip(X, Y):
        y[ydim+ypad:ydim+ypad+zdim] = np.random.randn(zdim)
        yield x, y


def gen_dataset(x_data, y_data, xdim, ydim, zdim, dim_tot, batch_size):
    N = x_data.shape[0]
    dataset = tf.data.Dataset.from_generator(
        gen, args=(x_data, y_data, xdim, ydim, zdim, dim_tot),
        output_types=('float32', 'float32'),
        output_shapes=(dim_tot, dim_tot)
    )
    
    dataset = dataset.shuffle(N).batch(batch_size, drop_remainder=True).repeat()
    return dataset
    
    
def fully_connected(n_dim, n_layer=3, n_hid=512, activation='relu'):
    x = tfk.Input((n_dim,))
    h = x
    for _ in range(n_layer):
        h = tfkl.Dense(n_hid, activation=activation)(h)
    log_s = tfkl.Dense(n_dim, activation='tanh')(h)
    t = tfkl.Dense(n_dim, activation='linear')(h)
    
    nn = tfk.Model(inputs=x, outputs=[log_s, t], name='nn')
    return nn


class TwoNVPCouplingLayers(tfkl.Layer):
    def __init__(self, inp_dim, n_hid_layer, n_hid_dim, name, shuffle_type):
        super(TwoNVPCouplingLayers, self).__init__(name=name)
        '''Implementation of Coupling layers in Ardizzone et al (2018)
        # Forward
        y1 = x1 * exp(s2(x2)) + t2(x2)
        y2 = x2 * exp(s1(x1)) + t1(x1)
        # Inverse
        x2 = (y2 - t1(y1)) * exp(-s1(y1))
        x1 = (y1 - t2(y2)) * exp(-s2(y2))
        '''
        self.inp_dim = inp_dim
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type
        self.nn1 = fully_connected(inp_dim//2, n_hid_layer, n_hid_dim)
        self.nn2 = fully_connected(inp_dim//2, n_hid_layer, n_hid_dim)
        self.idx = tf.Variable(list(range(self.inp_dim)),
                               shape=(self.inp_dim,),
                               trainable=False,
                               name=name+'_idx',
                               dtype='int64')
        if self.shuffle_type == 'random':
            self.idx.assign(tf.random.shuffle(self.idx))
        elif self.shuffle_type == 'reverse':
            self.idx.assign(tf.reverse(self.idx, axis=[0]))

    def call(self, x):
        x = self.shuffle(x, isInverse=False)
        x1, x2 = self.split(x)
        log_s2, t2 = self.nn2(x2)
        y1 = x1 * tf.math.exp(log_s2) + t2
        log_s1, t1 = self.nn1(y1)
        y2 = x2 * tf.math.exp(log_s1) + t1
        y = tf.concat([y1, y2], axis=-1)
        # Add loss
        self.log_det_J = log_s1 + log_s2
        self.add_loss(- tf.math.reduce_sum(self.log_det_J))
        return y

    def inverse(self, y):
        y1, y2 = self.split(y)
        log_s1, t1 = self.nn1(y1)
        x2 = (y2 - t1) * tf.math.exp(-log_s1)
        log_s2, t2 = self.nn2(x2)
        x1 = (y1 - t2) * tf.math.exp(-log_s2)
        x = tf.concat([x1, x2], axis=-1)
        x = self.shuffle(x, isInverse=True)
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
        dim = self.inp_dim
        x = tf.reshape(x, [-1, dim])
        return x[:, :dim//2], x[:, dim//2:]


class NVP(tfk.Model):
    def __init__(self, inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, name, shuffle_type='reverse'):
        super(NVP, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type
        self.AffineLayers = []
        for i in range(n_couple_layer):
            layer = TwoNVPCouplingLayers(
                inp_dim, n_hid_layer, n_hid_dim,
                name=f'Layer{i}', shuffle_type=shuffle_type)
            self.AffineLayers.append(layer)

    def call(self, x):
        '''Forward: data (x) --> latent (z); inference'''
        z = x
        for i in range(self.n_couple_layer):
            z = self.AffineLayers[i](z)
        return z

    def inverse(self, z):
        '''Inverse: latent (z) --> data (y); sampling'''
        x = z
        for i in reversed(range(self.n_couple_layer)):
            x = self.AffineLayers[i].inverse(x)
        return x


def NLL(y_true, y_pred):
    '''Negative Log-likelihood Loss'''
    return tf.math.reduce_sum(0.5*y_pred**2)
    
    
def MSE(y_true, y_pred):
    return tf.reduce_mean(tfk.losses.mean_squared_error(y_true, y_pred))


def MMD_multiscale(x, y):
    xx = tf.linalg.matmul(x, tf.transpose(x))
    yy = tf.linalg.matmul(y, tf.transpose(y))
    zz = tf.linalg.matmul(x, tf.transpose(y))

    rx = tf.broadcast_to(tf.linalg.diag_part(xx), xx.shape)
    ry = tf.broadcast_to(tf.linalg.diag_part(yy), yy.shape)

    dxx = tf.transpose(rx) + rx - 2.*xx
    dyy = tf.transpose(ry) + ry - 2.*yy
    dxy = tf.transpose(rx) + ry - 2.*zz

    XX = tf.zeros(xx.shape, dtype='float32')
    YY = tf.zeros(xx.shape, dtype='float32')
    XY = tf.zeros(xx.shape, dtype='float32')

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * 1/(a**2 + dxx)
        YY += a**2 * 1/(a**2 + dyy)
        XY += a**2 * 1/(a**2 + dxy)

    return tf.reduce_mean(XX + YY - 2.*XY)


class Trainer(tfk.Model):
    def __init__(self, model, x_dim, y_dim, z_dim, tot_dim, 
                 n_couple_layer, n_hid_layer, n_hid_dim, shuffle_type='reverse'):
        super(Trainer, self).__init__()
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.tot_dim = tot_dim
        self.x_pad_dim = tot_dim - x_dim
        self.y_pad_dim = tot_dim - (y_dim + z_dim)
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type

        self.w1 = 5.
        self.w2 = 1.
        self.w3 = 10.
        self.loss_factor = 1.
        self.loss_fit = MSE
        self.loss_latent = MMD_multiscale
        self.loss_latent_nll = NLL

    def train_step(self, data):
        x_data, y_data = data
        x = x_data[:, :self.x_dim]
        y = y_data[:, -self.y_dim:]
        z = y_data[:, :self.z_dim]
        y_short = tf.concat([z, y], axis=-1)

        # Forward loss
        with tf.GradientTape() as tape:
            y_out = self.model(x_data)    
            pred_loss = self.w1 * self.loss_fit(y_data[:,self.z_dim:], y_out[:,self.z_dim:]) # [zeros, y] <=> [zeros, yhat]
            output_block_grad = tf.concat([y_out[:,:self.z_dim], y_out[:, -self.y_dim:]], axis=-1) # take out [z, y] only (not zeros)
            latent_loss = self.w2 * self.loss_latent(y_short, output_block_grad) # [z, y] <=> [zhat, yhat]
            latent_loss += self.loss_latent_nll(z, y_out[:,:self.z_dim])
            forward_loss = pred_loss + latent_loss
        grads_forward = tape.gradient(forward_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_forward, self.model.trainable_weights))

        # Backward loss
        with tf.GradientTape() as tape:
            x_rev = self.model.inverse(y_data)
            rev_loss = self.w3 * self.loss_factor * self.loss_fit(x_rev, x_data)
        grads_backward = tape.gradient(rev_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_backward, self.model.trainable_weights)) 

        total_loss = forward_loss + latent_loss + rev_loss
        return {'total_loss': total_loss,
                'forward_loss': forward_loss,
                'latent_loss': latent_loss,
                'rev_loss': rev_loss}

    def test_step(self, data):
        x_data, y_data = data
        x = x_data[:, :self.x_dim]
        y = y_data[:, -self.y_dim:]
        z = y_data[:, :self.z_dim]
        y_short = tf.concat([z, y], axis=-1)
        
        # Forward loss
        y_out = self.model(x_data)    
        pred_loss = self.w1 * self.loss_fit(y_data[:,self.z_dim:], y_out[:,self.z_dim:]) # [zeros, y] <=> [zeros, yhat]
        output_block_grad = tf.concat([y_out[:,:self.z_dim], y_out[:, -self.y_dim:]], axis=-1) # take out [z, y] only (not zeros)
        latent_loss = self.w2 * self.loss_latent(y_short, output_block_grad) # [z, y] <=> [zhat, yhat]
        forward_loss = pred_loss + latent_loss
        
        # Backward loss
        x_rev = self.model.inverse(y_data)
        rev_loss = self.w3 * self.loss_factor * self.loss_fit(x_rev, x_data)
        
        total_loss = forward_loss + latent_loss + rev_loss
        return {'val_total_loss': total_loss,
                'val_forward_loss': forward_loss,
                'val_latent_loss': latent_loss,
                'val_rev_loss': rev_loss}


if __name__ == "__main__":
    inp_dim = 2
    n_couple_layer = 3
    n_hid_layer = 3
    n_hid_dim = 512

    model = NVP(inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='NVP')
    x = tfkl.Input(shape=(inp_dim,))
    model(x)
    model.summary()
    model.save_weights('test.h5')
    print('done')