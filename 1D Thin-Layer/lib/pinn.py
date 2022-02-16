import tensorflow as tf
from .layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for Fick's law equation.
    """

    def __init__(self, network):

        self.network = network
        self.grads = GradientLayer(self.network)

    def build(self):


        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        # initial condition input: (t=0, x)
        tx_ini = tf.keras.layers.Input(shape=(2,))
        tx_bnd = tf.keras.layers.Input(shape=(2,))

        tx_bnd_out = tf.keras.layers.Input(shape=(2,))




        # compute gradients
        c, dc_dt, dc_dx, d2c_dx2 = self.grads(tx_eqn)

        # equation output being zero
        c_eqn = dc_dt - d2c_dx2
        # initial condition output
        c_ini = self.network(tx_ini)
        # boundary condition output
        c_bnd = self.network(tx_bnd)

        c_out, dc_dt_out, dc_dx_out, d2c_dx2_out = self.grads(tx_bnd_out)

        c_bnd_out = dc_dx_out

        return tf.keras.models.Model(
            inputs=[tx_eqn, tx_ini, tx_bnd,tx_bnd_out], outputs=[c_eqn, c_ini, c_bnd,c_bnd_out])
