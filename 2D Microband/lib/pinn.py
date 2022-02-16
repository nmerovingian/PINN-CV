import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:



    def __init__(self, network, nu):


        self.network = network
        self.nu = nu
        self.grads = GradientLayer(self.network)



    def build(self):

        txy_dmn0 = tf.keras.layers.Input(shape=(3,))
        txy_dmn1 = tf.keras.layers.Input(shape=(3,))

        txy_ini = tf.keras.layers.Input(shape=(3,))

        txy_bnd1 = tf.keras.layers.Input(shape=(3,))
        txy_bnd4 = tf.keras.layers.Input(shape=(3,))


        txy_bnd0 = tf.keras.layers.Input(shape=(3,))
        txy_bnd2 = tf.keras.layers.Input(shape=(3,))
        txy_bnd3 = tf.keras.layers.Input(shape=(3,))

        c_ini = self.network(txy_ini)

        c, dc_dt, dc_dx,dc_dy, d2c_dx2,d2c_dy2 = self.grads(txy_dmn0)




        c_dmn = dc_dt -d2c_dy2 - d2c_dx2 

        c_2,dc_dt_2,dc_dx_2,dc_dy_2,d2c_dx2_2,d2c_dy2_2 = self.grads(txy_dmn1)

        c_dmn2 = dc_dt_2 - d2c_dx2_2 - d2c_dy2_2



        c1,dc_dt1,dc_dx1,dc_dy1,d2c_dx21,d2c_dy21 = self.grads(txy_bnd1)
        c4,dc_dt4,dc_dx4,dc_dy4,d2c_dx24,d2c_dy24 = self.grads(txy_bnd4)

        c_bnd1 =  dc_dy4
        c_bnd4 =  dc_dx1 


        # boundary condition output
        c_bnd0 = self.network(txy_bnd0)
        c_bnd2 = self.network(txy_bnd2)
        c_bnd3 = self.network(txy_bnd3)

        return tf.keras.models.Model(
            inputs=[txy_dmn0,txy_dmn1,txy_ini,txy_bnd1,txy_bnd4,txy_bnd0,txy_bnd2,txy_bnd3], outputs=[c_dmn,c_dmn2,c_ini,c_bnd1,c_bnd4,c_bnd0,c_bnd2,c_bnd3])