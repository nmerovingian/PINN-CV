import tensorflow as tf
from .layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for Fick's law equation.
    """

    def __init__(self, network):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
        """

        self.network = network
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for Burgers' equation.

        Returns:
            PINN model for the projectile motion with
                input: [ (t, x) relative to equation,
                         (t=0, x) relative to initial condition,
                         (t, x=bounds) relative to boundary condition ],
                output: [ c(t,x) relative to equation (must be zero),
                          c(t=0, x) relative to initial condition,
                          c(t, x=bounds) relative to boundary condition ]
        """

        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))

        tx_ini = tf.keras.layers.Input(shape=(2,))

        tx_bnd = tf.keras.layers.Input(shape=(2,))

        # compute gradients
        c, dc_dt, dc_dx, d2c_dx2 = self.grads(tx_eqn)

        # equation output being zero
        c_eqn = dc_dt - d2c_dx2
        # initial condition output
        c_ini = self.network(tx_ini)
        # boundary condition output
        c_bnd = self.network(tx_bnd)

        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[tx_eqn, tx_ini, tx_bnd], outputs=[c_eqn, c_ini, c_bnd])
