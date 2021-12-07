from re import I
import re
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
from matplotlib.patches import Rectangle
import tensorflow as tf 
import os
from scipy.signal import savgol_filter  

linewidth = 4
fontsize = 20

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs




def main(epochs=50,sigma=40,train=True,saving_directory = './Data',rElectrode=0.5,loss_weights = [1,1,1,1,1,1,1,1],alpha=1.0,lambda_ratio=2.0):
    """
    epochs: number of epoch for the training 
    sigma: dimensionless scan rate
    train: If true, always train a new neural network. Otherwise just 
    alpha: the decay of learning rate. If alpha = 1.0, no decay of learning rate. alpha<1 if decay of learning rate.

    
    """
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    # define the learning rate scheduler 
    def scheduler(epoch,lr):
        
        if epoch < 400:
            return lr
        else:
            return lr*alpha


    # number of training samples
    num_train_samples = int(5e6)
    # number of test samples
    num_test_samples = 1000

    D =  1.0


    file_name = f'sigma={sigma:.2E} epochs={epochs:.2E} n_train={num_train_samples:.2E} rElectrode={rElectrode:.2E} alpha={alpha:.2E} loss_weights={loss_weights} lambda_ratio = {lambda_ratio}'


    theta_i = 10.0 # start/end potential of scan 
    theta_v = -10.0 # reverse potential of scan


    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
    maxX = lambda_ratio* np.sqrt(maxT)  # the max diffusion length 



    # the training feature enforcing Fick's second law of diffusion in 2D
    txy_dmn0 = np.random.rand(num_train_samples,3)
    txy_dmn0[...,0] = txy_dmn0[...,0] * maxT
    txy_dmn0[...,1] = txy_dmn0[...,1] *maxX 
    txy_dmn0[...,2] = txy_dmn0[...,2] *(maxX+rElectrode)

    # a smaller domain near the surface of electrode
    txy_dmn1 = np.random.rand(num_train_samples,3)
    txy_dmn1[...,0] = txy_dmn1[...,0] * maxT
    txy_dmn1[...,1] = txy_dmn1[...,1] *1.0
    txy_dmn1[...,2] = txy_dmn1[...,2] *1.0 

    # boundary 1
    txy_bnd1 = np.random.rand(num_train_samples,3)
    txy_bnd1[...,0] = txy_bnd1[...,0] * maxT
    txy_bnd1[...,1] = 0.0 
    txy_bnd1[...,2] = txy_bnd1[...,2] * maxX + rElectrode

    # boundary 2 
    txy_bnd2 = np.random.rand(num_train_samples,3)
    txy_bnd2[...,0] = txy_bnd2[...,0] * maxT
    txy_bnd2[...,1] = txy_bnd2[...,1] * maxX 
    txy_bnd2[...,2] = (maxX + rElectrode)

    # boundary 3 
    txy_bnd3 = np.random.rand(num_train_samples,3)
    txy_bnd3[...,0] = txy_bnd3[...,0] * maxT
    txy_bnd3[...,1] = maxX
    txy_bnd3[...,2] = txy_bnd3[...,2] * (maxX + rElectrode) 

    #boundary 4
    txy_bnd4 = np.random.rand(num_train_samples,3)
    txy_bnd4[...,0] = txy_bnd4[...,0] * maxT
    txy_bnd4[...,1] = txy_bnd4[...,1] * maxX
    txy_bnd4[...,2] = 0.0

    # initial condition 
    txy_ini = np.random.rand(num_train_samples,3)
    txy_ini[...,0]  = 0.0
    txy_ini[...,1] = txy_ini[...,1] *maxX 
    txy_ini[...,2] = txy_ini[...,2] *(maxX+rElectrode) 


    # boundary 0, the electrode surface 
    txy_bnd0 = np.random.rand(num_train_samples,3) 
    txy_bnd0[...,0] = txy_bnd0[...,0] * maxT
    txy_bnd0[...,1] = 0.0
    txy_bnd0[...,2] = txy_bnd0[...,2] * rElectrode


    # creating training output 

    c_dmn = np.zeros((num_train_samples,1))
    c_dmn2 = np.zeros((num_train_samples,1))

    c_bnd0 = np.zeros((num_train_samples,1))
    c_bnd1 = np.zeros((num_train_samples,1))
    c_bnd2 = np.ones((num_train_samples,1))
    c_bnd3 = np.ones((num_train_samples,1))
    c_bnd4 = np.zeros((num_train_samples,1))

    c_ini = np.ones((num_train_samples,1))


    for i in range(num_train_samples):
        if txy_bnd0[i,0] < maxT/2.0 and txy_bnd0[i,1] < 0.01 and txy_bnd0[i,2]<=1.0:
            c_bnd0[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*txy_bnd0[i,0])))

        elif txy_bnd0[i,0] > maxT/2.0 and txy_bnd0[i,1] < 0.01 and txy_bnd0[i,2]<=1.0:
            c_bnd0[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(txy_bnd0[i,0]-maxT/2.0))))
        else:
            c_bnd0[i] = 1.0



    x_train = [txy_dmn0,txy_dmn1,txy_ini,txy_bnd1,txy_bnd4,txy_bnd0,txy_bnd2,txy_bnd3]
    y_train = [c_dmn,c_dmn2,c_ini,c_bnd1,c_bnd4,c_bnd0,c_bnd2,c_bnd3]



    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, D).build()

    pinn.compile(optimizer='Adam',loss='mse',loss_weights=loss_weights)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # If train = True, fit the NN, else, use saved weights 
    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =500,verbose=2,callbacks=[callback])
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =500,verbose=2)
            pinn.save_weights(f'./weights/weights {file_name}.h5')


    
    time_sects = [maxT/2.0]

    for index,time_sect in enumerate(time_sects):
        txy_test = np.zeros((int(num_test_samples**2),3))
        txy_test[...,0] = time_sect
        x_flat = np.linspace(0,maxX,num_test_samples)
        y_flat = np.linspace(0,maxX+rElectrode,num_test_samples)
        x,y = np.meshgrid(x_flat,y_flat)
        txy_test[...,1] = x.flatten()
        txy_test[...,2] = y.flatten()

        c = network.predict(txy_test)

        c = c.reshape(x.shape)

        plt.figure()
        plt.pcolormesh(x,y,c,shading='auto')
        plt.title(f'time={time_sect:.2f}')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('c(t,x,y)')
        cbar.mappable.set_clim(0, 1)
        plt.savefig(f'{saving_directory}/{file_name} t={index}.png')
        plt.close('all')
    
    
    time_steps = np.linspace(0.0,maxT,num=3000)
    cv_flat = np.where(time_steps<maxT/2.0,theta_i-sigma*time_steps,theta_v+sigma*(time_steps-maxT/2.0))

    
    # extract the cyclic voltammogram 
    fluxes = np.zeros_like(time_steps)
    for index, time_step in enumerate(time_steps):
        x_flat = np.linspace(0,1e-5,25)
        y_flat = np.linspace(0,rElectrode,num=500)
        txy_test = np.zeros((int(len(x_flat)*len(y_flat)),3))
        txy_test[...,0] = time_step
        x,y = np.meshgrid(x_flat,y_flat)
        txy_test[...,1] = x.flatten()
        txy_test[...,2] = y.flatten()

        x_i = x_flat[1] -x_flat[0]
        y_i = y_flat[1] - y_flat[0]
        c = network.predict(txy_test)
        c = c.reshape(x.shape)

        J = - sum((c[:,3] - c[:,0])/ (3*x_i) * y_i)



        fluxes[index] = J



    df = pd.DataFrame({'Potential':cv_flat,'Flux':fluxes,})
    df['Flux'] /= rElectrode

    df['Flux'] = savgol_filter(df['Flux'],21,3)

    df.to_csv(f'{saving_directory}/{file_name}.csv',index=False)

    
    fig, ax = plt.subplots()
    df = pd.read_csv(f'{saving_directory}/{file_name}.csv')
    df.plot(x='Potential',y='Flux',ax=ax)
    fig.savefig(f'{saving_directory}/{file_name}.png')

    



    tf.keras.backend.clear_session()




if __name__ == "__main__":

    for epochs in [1600]:
        for sigma in [4]:
            for loss_weights in [[1,1,1,1,1,1,1,1]]:
                for rElectrode in [0.5]:
                    for alpha in [0.99]:
                        for lambda_ratio in [2.0]:
                            main(epochs=epochs,sigma=sigma,train=False,rElectrode=rElectrode,alpha=alpha,loss_weights=loss_weights,lambda_ratio=lambda_ratio)  