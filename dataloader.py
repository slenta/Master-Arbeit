from matplotlib.pyplot import axis, plot
import pylab as plt
import numpy as np
import netCDF4 as nc
import h5py
import torch
from torch._C import dtype
#from torch._C import float32
#from torch._C import float32
#from torch._C import double
from torch.utils import data
from torch.utils.data import Dataset
import xarray as xr
from torchvision.utils import make_grid
from torchvision.utils import save_image
from image import unnormalize




#dataloader and dataloader

def preprocessing(new_im_size, path, name, year, type, plot):
    
    ds = xr.load_dataset(path + name + year + '.nc', decode_times=False)

    #extract the variables from the file
    if type == 'mask':
        sst = ds.tho.values[:, 0, :, :]
        n = sst.shape
        
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    if np.isnan(sst[i, j, k]) == True:
                        sst[i, j, k] = 0
                    else:
                        sst[i, j, k] = 1
        
 
        rest = np.zeros((n[0], new_im_size - n[1], n[2]))
        sst = np.concatenate((sst, rest), axis=1)
        n = sst.shape
        rest2 = np.zeros((n[0], n[1], new_im_size - n[2]))
        sst_new = np.concatenate((sst, rest2), axis=2)

        n = sst_new.shape
        #create new h5 file with symmetric ssts
        f = h5py.File(path + name + year + '.hdf5', 'w')
        dset1 = f.create_dataset('tos_sym', (n[0], n[1], n[2]), dtype = 'float32', data = sst_new)
        f.close()

    if type == 'image':
        sst = ds.thetao.values[:, 0, :, :]
        x = np.isnan(sst)
        n = sst.shape
        sst[x] = 0

        rest = np.zeros((n[0], new_im_size - n[1], n[2]))
        sst = np.concatenate((sst, rest), axis=1)
        n = sst.shape
        rest2 = np.zeros((n[0], n[1], new_im_size - n[2]))
        sst_new = np.concatenate((sst, rest2), axis=2)
        n = sst_new.shape 
        #create new h5 file with symmetric ssts
        f = h5py.File(path + name + year + '.hdf5', 'w')
        dset1 = f.create_dataset('tos_sym', (n[0], n[2], n[2]), dtype = 'float32',data = sst_new)
        f.close()

    #plot ssts in 2d plot
    if plot == True:
        pixel_plot = plt.figure()
        pixel_plot = plt.imshow(sst_new[1], vmin = -5, vmax = 5)
        plt.colorbar(pixel_plot)
        plt.savefig('../Asi_maskiert/pdfs/' + name + '.pdf')
        plt.show()
        
#preprocessing(128, '../Asi_maskiert/original_masks/', 'Maske_', '2001_2020_newgrid', type='mask', plot=False)
#preprocessing(128, '../Asi_maskiert/original_image/', 'Image_', 'r10_11_newgrid', type='image', plot=False)



class MaskDataset(Dataset):

    def __init__(self, depth, in_channels, year, im_year, mode):
        super(MaskDataset, self).__init__()

        self.image_path = '../Asi_maskiert/original_image/'
        self.mask_path = '../Asi_maskiert/original_masks/'
        self.image_name = 'Image_'
        self.mask_name = 'Maske_'
        self.image_year = im_year
        self.year = year
        self.mode = mode
        self.in_channels = in_channels
        self.depth = depth

    def __getitem__(self, index):

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(self.image_path + self.image_name + self.image_year + '.hdf5', 'r')
        f_mask = h5py.File(self.mask_path + self.mask_name + self.year + '.hdf5', 'r')

        #extract sst data/mask data
        image = f_image.get('tos_sym')
        mask = f_mask.get('tos_sym')
        mask = np.repeat(mask, 5, axis=0)

        n = image.shape
        m = mask.shape

        im_new = []

        if self.mode == 'train':
            for i in range(n[0]):
                if i%5 >= 1:
                    im_new.append(image[i])
        elif self.mode == 'val':
            mask = mask[:8]
            for i in range(n[0]):
                if i%5 == 0:
                    im_new.append(image[i])
            im_new = im_new[:8]

        im_new = np.array(im_new)
        np.random.shuffle(im_new)
        np.random.shuffle(mask)

        #convert to pytorch tensors
        if self.depth==True:
            im_new = torch.from_numpy(im_new[index, :self.in_channels, :, :])
            mask = torch.from_numpy(mask[index, :self.in_channels, :, :])
        elif self.depth ==False:
            mask = torch.from_numpy(mask[index, :, :])
            im_new = torch.from_numpy(im_new[index, :, :])
            #Repeat to fit input channels
            mask = mask.repeat(3, 1, 1)
            im_new = im_new.repeat(3, 1, 1)

        return mask*im_new, mask, im_new

    def __len__(self):
        
        mi, ma, im_new = self.__getitem__(0)
        n = im_new.shape
        length = n[0]

        return length


class SpecificValDataset():
    
    def __init__(self, timestep, year):
        super(SpecificValDataset, self).__init__()

        self.image_path = '../Asi_maskiert/original_image/'
        self.mask_path = '../Asi_maskiert/original_masks/'
        self.image_name = 'Observation_'
        self.mask_name = 'Observation_'
        self.year = year
        self.timestep = timestep

    def __getitem__(self, index):

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(self.image_path + self.image_name + self.year + '.hdf5', 'r')
        f_mask = h5py.File(self.mask_path + self.mask_name + self.year + '.hdf5', 'r')
        f_gt = h5py.File(self.image_path + 'Image_2020.hdf5', 'r')

        #extract sst data/mask data
        image = f_image.get('tos_sym')
        mask = f_mask.get('tos_sym')
        gt = f_gt.get('tos_sym')[self.timestep, :, :]

        #repeat to insert time dimension
        image = np.repeat(image, 16, axis=0)
        mask = np.repeat(mask, 16, axis=0)
        gt = np.expand_dims(gt, axis=0)
        gt = np.repeat(gt, 16, axis=0)

        #convert to pytorch tensors
        im_new = torch.from_numpy(image[index, :, :])
        mask = torch.from_numpy(mask[index, :, :])
        gt = torch.from_numpy(gt[index, :, :])

        #bring into right shape   
        mask = mask.repeat(3, 1, 1)
        im_new = im_new.repeat(3, 1, 1)
        gt = gt.repeat(3, 1, 1)


        return im_new, mask, gt



#dataset1 = SpecificValDataset(12*27 + 11, '11_1985')
#mi, m, i = dataset1[0]

#dataset1 = MaskDataset(7, '2020_newgrid', '3d_1958_2020_newgrid', 'train')
#mi, m, i, = dataset1[3]
#print(mi.shape, m.shape, i.shape)

#f_mask = h5py.File('../Asi_maskiert/original_masks/Maske_2004_2020.hdf5', 'r')
#mask = f_mask.get('tos_sym')
