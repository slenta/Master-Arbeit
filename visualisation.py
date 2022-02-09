import h5py
from isort import file
from matplotlib import image
import numpy as np
import pylab as plt
import torch
import xarray as xr
import cdo
cdo = cdo.Cdo()

def vis_single(timestep, path, name, argo_state, type, title):

    #df = xr.load_dataset(path + name + '.nc', decode_times=False)
    f = h5py.File(path + name + '.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinentmaske.hdf5', 'r')
    
    continent_mask = fm.get('tos_sym')
    cm = np.array(continent_mask[:220, :])

    output_data = f.get(type)
    image_data = f.get('image')
    #sst = df.thetao.values
    #x = np.isnan(sst)
    #sst[x] = -15
    #sst = sst[timestep, 10, :, :] #+ (1-cm)*99
    sst = output_data[timestep, 0, :, :]
    image = image_data[timestep, 0, :, :]

    if type=='output':
        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
        plt.colorbar()
        plt.savefig('../Asi_maskiert/pdfs/' + name + argo_state + '.pdf')
        plt.show()
    elif type=='mask':
        sst = sst * image
        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
        plt.colorbar()
        plt.savefig('../Asi_maskiert/pdfs/' + name + type + argo_state + '.pdf')
        plt.show()

def visualisation(iter):
    
    f = h5py.File('../Asi_maskiert/results/images/r1011_shuffle_newgrid/short_val/part_1/test_' + iter + '.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinentmaske.hdf5', 'r')
    
    continent_mask = fm.get('tos_sym')
    image_data = f.get('image')[5, 2, :, :]
    mask_data = f.get('mask')[5, 2,:, :]
    output_data = f.get('output')[5, 2,:, :]

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
    #plt.savefig('../Asi_maskiert/results/images/part_1/test_' + iter + '.pdf')
    plt.show()

#visualisation('1350000')

#vis_single(753, '../Asi_maskiert/original_image/', 'Image_3d_1958_2020', 'Global Subsurface Temperature Assimilation October 2020')
vis_single(7, '../Asi_maskiert/results/images/r1011_shuffle_newgrid/short_val/Maske_1970_1985/', 'test_700000', 'pre_argo', 'mask', 'Pre-Argo Mask')