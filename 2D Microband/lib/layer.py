import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):


    def __init__(self, model, **kwargs):


        self.model = model
        super().__init__(**kwargs)

    def call(self, x):


        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(x)
                c = self.model(x)
            dc_dtxy = gg.batch_jacobian(c, x)
            dc_dt = dc_dtxy[..., 0]
            dc_dx = dc_dtxy[..., 1]
            dc_dy = dc_dtxy[..., 2]

        d2c_dy2 = g.batch_jacobian(dc_dy,x)[...,2]
        d2c_dx2 = g.batch_jacobian(dc_dx, x)[..., 1]

        return c, dc_dt, dc_dx,dc_dy, d2c_dx2,d2c_dy2


class BoundaryGradientLayer(tf.keras.layers.Layer):
    """
    Custome layer to compute the 1st derivative for no flux boundary conditions
    
    """


    def __init__(self,model,**kwargs):
        self.model = model

        super().__init__(**kwargs)


    def call(self,x):
        """
        Computing 1st derivative 

        X: input variable
        return 1 st derivative 
        """


        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            c = self.model(x)

        dc_dtxy = g.batch_jacobian(c,x)
        dc_dt = dc_dtxy[...,0]
        dc_dx = dc_dtxy[...,1]
        dc_dy = dc_dtxy[...,2]

        return c,dc_dt,dc_dx,dc_dy