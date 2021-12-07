import numpy as np 
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras import callbacks
from tensorflow.python.keras.callbacks import Callback
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler




def main(epochs=50,sigma=40,train=True,plot_cv=True,saving_directory="./Data",alpha=1.0):
    """
    epochs: number of epoch for the training 
    sigma: dimensionless scan rate
    train: If true, always train a new neural network. If false, use the existing weights. If weights does not exist, start training 
    plot_cv: plot the resulting CV after training 
    saving directory: where data is saved 
    """
    def schedule(epoch,lr):
        # a learning rate scheduler 
        if epoch<=400:
            return lr 
        else:
            lr *= alpha
            return lr
    # saving directory is where data(voltammogram, concentration profile etc is saved)
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)
    
    #weights folder is where weights is saved
    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    # number of training samples
    num_train_samples = int(1e6)
    # number of test samples
    num_test_samples = 1000
    # Dimensionless Diffusion coefficient. Since it is always 1, not very relevant in this problem
    D =  1.0

    # prefix or suffix of files 
    file_name = f'sigma={sigma:.2E} epochs={epochs:.2E} n_train={num_train_samples:.2E}'

    theta_i = 10.0 # start/end potential of scan 
    theta_v = -10.0 # reverse potential of scan


    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
    maxX = 6.0 * np.sqrt(maxT)  # the max diffusion length 


    # the training feature enoforcing fick's second law 
    # two fick's second law of diffusion scan in the X and Y direction 
    txy_eqn0 = np.random.rand(num_train_samples,3)
    txy_eqn0[:,0] = txy_eqn0[:,0] * maxT
    txy_eqn0[:,1] = txy_eqn0[:,1] * maxX + 1.0
    txy_eqn0[:,2] = txy_eqn0[:,2] * (maxX+1.0)

    txy_eqn1 = np.random.rand(num_train_samples,3)
    txy_eqn1[:,0] = txy_eqn1[:,0] * maxT
    txy_eqn1[:,1] = txy_eqn1[:,1] * (maxX+1.0)
    txy_eqn1[:,2] = txy_eqn1[:,2] * maxX + 1.0

    # the initial condition
    txy_ini = np.random.rand(num_train_samples,3)
    txy_ini[:,0] = 0.0
    txy_ini[:,1] = txy_ini[:,1] * (maxX+1.0)
    txy_ini[:,2] = txy_ini[:,2] * (maxX+1.0)


    #the boundary conditions
    txy_bnd_Nernst = np.random.rand(num_train_samples,3)
    txy_bnd_Nernst[:,0] = txy_bnd_Nernst[:,0] * maxT
    txy_bnd_Nernst[:,1] = txy_bnd_Nernst[:,1] * 1.0 
    txy_bnd_Nernst[:,2] = txy_bnd_Nernst[:,2] * 1.0

    # no flux boundary condition
    txy_bnd_1 = np.random.rand(num_train_samples,3)
    txy_bnd_1[:,0] = txy_bnd_1[:,0] * maxT
    txy_bnd_1[:,1] = txy_bnd_1[:,1] * maxX + 1.0
    txy_bnd_1[:,2] = 0.0


    # no flux boundary condition
    txy_bnd_2 = np.random.rand(num_train_samples,3)
    txy_bnd_2[:,0] = txy_bnd_2[:,0] * maxT
    txy_bnd_2[:,1] = 0.0
    txy_bnd_2[:,2] = txy_bnd_2[:,2] * maxX + 1.0

    # fixed concentration boundary condition        
    txy_bnd_3 = np.random.rand(num_train_samples,3)
    txy_bnd_3[:,0] = txy_bnd_3[:,0] * maxT
    txy_bnd_3[:,1] = txy_bnd_3[:,1] * (maxX+1.0)
    txy_bnd_3[:,1] = maxX+1.0

    # fixed concentration boundary condition
    txy_bnd_4 = np.random.rand(num_train_samples,3)
    txy_bnd_4[:,0] = txy_bnd_4[:,0] * maxT
    txy_bnd_4[:,1] = maxX+1.0
    txy_bnd_4[:,2] = txy_bnd_4[:,2] * (maxX+1.0)






    # output of each condition 

    c_eqn0 = np.zeros((num_train_samples,1))
    c_eqn1 = np.zeros((num_train_samples,1))


    c_ini = np.ones((num_train_samples,1))


    c_bnd_Nernst = np.zeros((num_train_samples,1))
    for i in range(num_train_samples):
        if txy_bnd_Nernst[i,0] < maxT/2.0:
            c_bnd_Nernst[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*txy_bnd_Nernst[i,0])))
        else:
            c_bnd_Nernst[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(txy_bnd_Nernst[i,0]-maxT/2.0))))

    c_bnd_1 = np.zeros((num_train_samples,1))
    c_bnd_2 = np.zeros((num_train_samples,1))

    c_bnd_3 = np.ones((num_train_samples))
    c_bnd_4 = np.ones((num_train_samples))





    x_train = [txy_eqn0,txy_eqn1,txy_ini,txy_bnd_Nernst,txy_bnd_1,txy_bnd_2,txy_bnd_3,txy_bnd_4]
    y_train = [c_eqn0,c_eqn1,c_ini,c_bnd_Nernst,c_bnd_1,c_bnd_2,c_bnd_3,c_bnd_4]


    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, D).build()

    # the loss weight of each loss componentan can be varied
    pinn.compile(optimizer='Adam',loss='mse',loss_weights=[1,1,1,1,1,1,1,1])

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    # If train = True, fit the NN, else, use saved weights 
    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =500,verbose=2,callbacks=[lr_scheduler_callback])
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =500,verbose=2,callbacks=[lr_scheduler_callback])
            pinn.save_weights(f'./weights/weights {file_name}.h5')
    
    
    # generate cylic voltammogram
    time_steps = np.linspace(0.0,maxT,num=num_test_samples)
    cv_flat = np.where(time_steps<maxT/2.0,theta_i-sigma*time_steps,theta_v+sigma*(time_steps-maxT/2.0))
    fluxes_right = np.zeros_like(time_steps)
    fluxes_top = np.zeros_like(time_steps)
    for index, time_step in enumerate(time_steps):
        # right and top edge of the particle is active 
        x_flat_right = np.linspace(1.0,1.0+1e-5,num=30)
        y_flat_right = np.linspace(0,1,num=500)
        txy_right = np.zeros((int(len(x_flat_right)*len(y_flat_right)),3))
        txy_right[...,0] = time_step
        x,y = np.meshgrid(x_flat_right,y_flat_right)
        txy_right[...,1] = x.flatten()
        txy_right[...,2] = y.flatten()

        x_i = x_flat_right[1] - x_flat_right[0]
        y_i = y_flat_right[1] - y_flat_right[0]
        c = network.predict(txy_right)
        c = c.reshape(x.shape)
        J = - sum((c[:,20] - c[:,0])/ (20*x_i) * y_i)

        fluxes_right[index] = J

        x_flat_top = np.linspace(0,1,num=500)
        y_flat_top = np.linspace(1.0,1.0+1e-5,num=30)
        txy_top = np.zeros((int(len(x_flat_top)*len(y_flat_top)),3))
        txy_top[...,0] = time_step
        x,y = np.meshgrid(x_flat_top,y_flat_top)
        txy_top[...,1] = x.flatten()
        txy_top[...,2] = y.flatten()

        x_i = x_flat_top[1] - x_flat_top[0]
        y_i = y_flat_top[1] - y_flat_top[0]
        c = network.predict(txy_top)
        c = c.reshape(x.shape)
        J = - sum((c[20,:] - c[0,:])/(20*y_i) * x_i)

        fluxes_top[index] = J

    df = pd.DataFrame({'Potential':cv_flat,"Right Edge":fluxes_right,"Top Edge":fluxes_top})
    df['Fluxes'] = df['Right Edge'] + df['Top Edge']

    df.to_csv(f'{saving_directory}/{file_name}.csv',index=False)
    
    # plot the cyclic voltammogram
    if plot_cv:
        fig,ax = plt.subplots(figsize=(8,4.5))
        df.plot(x='Potential',y=['Right Edge','Top Edge'],ax=ax)
        ax.legend()
        fig.savefig(f'{saving_directory}/{file_name}.png')

    





    tf.keras.backend.clear_session()


if __name__ =='__main__':
    # set train to true if you want to start training afresh
    main(epochs=400,sigma=40,train=False,alpha=1.0)

  