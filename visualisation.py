from cProfile import label
import h5py
from isort import file
from matplotlib import image
import numpy as np
import pylab as plt
import torch
import xarray as xr
from mpl_toolkits.mplot3d import axes3d
import cdo
cdo = cdo.Cdo()

def vis_single(timestep, path, name, argo_state, type, param, title):

    if type=='output':
        f = h5py.File(path + name + '.hdf5', 'r')

        output = f.get('output')[timestep, 0, :, :]
        image = f.get('image')[timestep, 0, :, :]
        mask = f.get('mask')[timestep, 0, :, :]
        masked = mask * image
        outputcomp = mask*image + (1 - mask)*output

        plt.figure(figsize=(6, 6))
        plt.title(title)
        #plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
        plt.imshow(masked, vmin = -5, vmax = 30, cmap='jet')
        plt.colorbar(label='Temperature in °C')
        plt.savefig('../Asi_maskiert/pdfs/' + name + param + argo_state + '.pdf')
        plt.show()

    elif type=='image':
        df = xr.load_dataset(path + name + '.nc', decode_times=False)

        sst = df.thetao.values
        sst = sst[0, 10, :, :]
        x = np.isnan(sst)
        sst[x] = -15
        
        plt.figure(figsize=(6, 4))
        plt.title(title)
        #plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
        plt.imshow(sst, vmin = -5, vmax = 30)
        plt.colorbar(label='Temperature in °C')
        plt.savefig('../Asi_maskiert/pdfs/' + name + argo_state + '.pdf')
        plt.show()
    
    elif type=='mask':
        df = xr.load_dataset(path + name + '.nc', decode_times=False)
        sst = df.tho.values
        sst = sst[timestep, 10, :, :]

        x = np.isnan(sst)
        sst[x] = -15

        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
        plt.colorbar()
        plt.savefig('../Asi_maskiert/pdfs/' + name + type + argo_state + '.pdf')
        plt.show()

    elif type=='3d':
        df = xr.load_dataset(path + name + '.nc', decode_times=False)

        sst = df.thetao.values[timestep, :, :, :]
        plt.figure()
        plt.title(title)
        ax = plt.subplot(111, projection='3d')

        z = sst[0, :, :]
        x = df.x.values
        y = df.y.values

        x = np.concatenate((np.zeros(17), x))
        print(x.shape, y.shape)
        scatter = ax.scatter(x, y, z, c=z, alpha=1)
        plt.colorbar(scatter, label='Temperatur in °C')

        plt.show()




def visualisation(iter):
    
    f = h5py.File('../Asi_maskiert/results/images/r8_16_newgrid/Maske_2020_newgrid/test_' + iter + '.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinentmaske.hdf5', 'r')
    
    continent_mask = fm.get('tos_sym')
    image_data = f.get('image')[0, 2, :, :]
    mask_data = f.get('mask')[0, 2,:, :]
    output_data = f.get('output')[0, 2,:, :]

    mask = torch.from_numpy(mask_data)
    output = torch.from_numpy(output_data)
    image = torch.from_numpy(image_data)
    outputcomp = mask*image + (1 - mask)*output
    Mse = np.mean((np.array(outputcomp) - np.array(image))**2)
    print(Mse)

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 4, 1)
    plt.title('Masked Image')
    im1 = plt.imshow(image * mask, vmin=-5, vmax=30, cmap='jet', aspect='auto')
    plt.subplot(1, 4, 2)
    plt.title('NN Output')
    im2 = plt.imshow(outputcomp, cmap = 'jet', vmin=-5, vmax=30, aspect = 'auto')
    plt.subplot(1, 4, 3)
    plt.title('Original Assimilation Image')
    im3 = plt.imshow(image, cmap='jet', vmin=-5, vmax=30, aspect='auto')
    plt.subplot(1, 4, 4)
    plt.title('Error')
    im5 = plt.imshow(image - output, vmin=-2, vmax=2, cmap='jet', aspect='auto')
    #plt.savefig('../Asi_maskiert/results/images/r1011_shuffle_newgrid/short_val/Maske_1970_1985/test_' + iter + '.pdf')
    plt.show()

#visualisation('900000')

#vis_single(753, '../Asi_maskiert/original_image/', 'Image_3d_newgrid', 'Argo-era', '3d', 'North Atlantic Assimilation October 2020')
#vis_single(9, '../Asi_maskiert/original_masks/', 'Maske_2020_newgrid', 'pre-Argo-era', '3d', 'North Atlantic Observations October 2020')

vis_single(0, '../Asi_maskiert/results/images/r8_16_newgrid/Maske_2020_newgrid/', 'test_900000', 'argo-era', 'output', 'mask', 'Argo-Era Mask')