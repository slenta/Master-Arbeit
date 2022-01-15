import h5py
from matplotlib import image
import numpy as np
import pylab as plt
import torch




def visualisation(iter):
    
    f = h5py.File('../Asi_maskiert/results/images/part_1/test_' + iter + '.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinentmaske.hdf5', 'r')
    
    continent_mask = fm.get('tos_sym')
    image_data = f.get('image')[2, 0, :, :]
    mask_data = f.get('mask')[2, 0,:, :]
    output_data = f.get('output')[2, 0, :, :] * continent_mask

    mask = torch.from_numpy(mask_data)
    output = torch.from_numpy(output_data)
    image = torch.from_numpy(image_data)

    Mse = np.mean((output_data - image_data)**2)
    print(Mse)

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 4, 1)
    plt.title('Masked Image')
    im1 = plt.imshow(image * mask, vmin=0, vmax=40, cmap='jet', aspect='auto')
    plt.subplot(1, 4, 2)
    plt.title('NN Output')
    im2 = plt.imshow(output, cmap = 'jet', vmin=-10, vmax=40, aspect = 'auto')
    plt.subplot(1, 4, 3)
    plt.title('Original Assimilation Image')
    im3 = plt.imshow(image, cmap='jet', vmin=-10, vmax=40, aspect='auto')
    plt.subplot(1, 4, 4)
    plt.title('Error')
    im5 = plt.imshow(image - output, vmin=-3, vmax=3, cmap='jet', aspect='auto')
    plt.savefig('../Asi_maskiert/results/images/part_1/test_' + iter + '.pdf')
    plt.show()

visualisation('350000')
