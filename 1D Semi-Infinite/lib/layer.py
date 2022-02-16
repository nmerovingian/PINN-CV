import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for Fick's law of diffusion

    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, x):
        """
        Computing 1st and 2nd derivatives for Fick's equation.

        Args:
            x: input variable.

        Returns:
            model output, 1st and 2nd derivatives.
        """

        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                u = self.model(x)
            dc_dtx = gg.batch_jacobian(u, x)
            dc_dt = dc_dtx[..., 0]
            dc_dx = dc_dtx[..., 1]
        d2c_dx2 = g.batch_jacobian(dc_dx, x)[..., 1]
        return u, dc_dt, dc_dx, d2c_dx2
