import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import math
import tensorflow as tf



def main(epochs=600,sigma=40,train=True):
    """
    Using PINN to solve 1D linear diffusion equation
    """

    # number of training samples
    n_train_samples = 100000
    # number of test samples
    num_test_samples = 1000

    theta_i = 10.0
    theta_v = -10.0
    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
    maxX = 6.0 * np.sqrt(maxT)  # the max diffusion length 

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network).build()

    # create training input
    tx_eqn = np.random.rand(n_train_samples, 2)
    tx_eqn[...,0] = tx_eqn[...,0] * maxT         
    tx_eqn[..., 1] = tx_eqn[..., 1]*maxX               

    tx_ini = np.random.rand(n_train_samples, 2)  
    tx_ini[..., 0] = 0
    tx_ini[...,1] = tx_ini[...,1]*maxX                                    
    tx_bnd = np.random.rand(n_train_samples, 2)         
    tx_bnd[...,0] = tx_bnd[...,0] *maxT
    tx_bnd[..., 1] =  np.round(tx_bnd[..., 1])*maxX   


    # create training output
    c_eqn = np.zeros((n_train_samples, 1))              
    c_ini = np.ones((n_train_samples,1))    

    c_bnd=np.ones((n_train_samples,1))

    # apply Nernst equation in boundary conditions
    for i in range(n_train_samples):
        if tx_bnd[i,0] < maxT/2.0 and tx_bnd[i,1]<1e-3:
            c_bnd[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*tx_bnd[i,0])))
        elif tx_bnd[i,0] >= maxT/2.0 and tx_bnd[i,1]<1e-3:
            c_bnd[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(tx_bnd[i,0]-maxT/2.0))))
        else:
            c_bnd[i] = 1.0
    

    # train the model using Adama
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [ c_eqn,  c_ini,  c_bnd]

    pinn.compile(optimizer='adam',loss='mse')

    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=1)
        pinn.save_weights(f'./weights/weights sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
    else:
        try:
            print(f'weights sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
            pinn.load_weights(f'./weights/weights sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
        except:
            print('weights does not exist\n start training')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=1)
            pinn.save_weights(f'./weights/weights sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')


    
    # predict c(t,x) distribution
    t_flat = np.linspace(0, maxT, num_test_samples)
    cv_flat = np.where(t_flat<maxT/2.0,theta_i-sigma*t_flat,theta_v+sigma*(t_flat-maxT/2.0))
    x_flat = np.linspace(0, maxX, num_test_samples) 
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    c = network.predict(tx, batch_size=num_test_samples)
    c = c.reshape(t.shape)
    x_i = x_flat[1]
    flux = -(c[1,:] - c[0,:])/x_i
    df = pd.DataFrame({'Potential':cv_flat,'Flux':flux})
    df.to_csv(f'Voltammogram scan sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.csv',index=False)
    fig, ax = plt.subplots()
    ax.plot(cv_flat,flux,label='PINN prediction')
    ax.axhline(-0.446*math.sqrt(sigma),label='R-S equation',ls='--',color='r')
    ax.axvline(-1.109,label='Expected Forward Scan Potential',ls='--',color='k')

    ax.set_xlabel('Potential, theta')
    ax.set_ylabel('Flux,J')
    ax.legend()
    fig.savefig(f'Voltmmorgam sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.png')
    

    

    tf.keras.backend.clear_session()



if __name__ == "__main__":
    # set train to True if you prefer to start fresh training.

    for sigma in [40]:
        main(epochs=600,sigma=sigma,train=False)