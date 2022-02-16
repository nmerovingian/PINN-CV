import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:


    def __init__(self, network,D):


        self.network = network
        self.D = D
        self.grads = GradientLayer(self.network)
        self.boundaryGrad = BoundaryGradientLayer(self.network)



    def build(self):

        txy_eqn0 = tf.keras.layers.Input(shape=(3,))
        txy_eqn1 = tf.keras.layers.Input(shape=(3,))

        txy_ini = tf.keras.layers.Input(shape=(3,))

        txy_bnd_Nernst = tf.keras.layers.Input(shape=(3,))
        txy_bnd_1 = tf.keras.layers.Input(shape=(3,))
        txy_bnd_2 = tf.keras.layers.Input(shape=(3,))
        txy_bnd_3 = tf.keras.layers.Input(shape=(3,))
        txy_bnd_4 = tf.keras.layers.Input(shape=(3,))

        ceqn0, dc_dt_eqn0, dc_dx_eqn0,dc_dy_eqn0, d2c_dx2_eqn0,d2c_dy2_eqn0 = self.grads(txy_eqn0)
        c_eqn0 = dc_dt_eqn0 - d2c_dx2_eqn0 - d2c_dy2_eqn0

        ceqn1, dc_dt_eqn1, dc_dx_eqn1,dc_dy_eqn1, d2c_dx2_eqn1,d2c_dy2_eqn1 = self.grads(txy_eqn1)
        c_eqn1 = dc_dt_eqn1 - d2c_dx2_eqn1 - d2c_dy2_eqn1

        c_ini = self.network(txy_ini)

        c_bnd_Nernst = self.network(txy_bnd_Nernst)

        cbnd1,dc_dt_bnd1,dc_dx_bnd1,dc_dy_bnd1 = self.boundaryGrad(txy_bnd_1)
        c_bnd_1 = dc_dy_bnd1

        cbnd2, dc_dt_bnd2,dc_dx_bnd2,dc_dy_bnd2 = self.boundaryGrad(txy_bnd_2)
        c_bnd_2 = dc_dx_bnd2

        c_bnd_3 = self.network(txy_bnd_3)
        c_bnd_4 = self.network(txy_bnd_4)







        return tf.keras.models.Model(
            inputs=[txy_eqn0,txy_eqn1,txy_ini,txy_bnd_Nernst,txy_bnd_1,txy_bnd_2,txy_bnd_3,txy_bnd_4], outputs=[c_eqn0,c_eqn1,c_ini,c_bnd_Nernst,c_bnd_1,c_bnd_2,c_bnd_3,c_bnd_4])