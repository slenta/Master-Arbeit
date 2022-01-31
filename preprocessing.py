
import numpy as np
import matplotlib.pylab as plt
from sympy import N
import xarray as xr
import config as cfg
import h5py


class preprocessing():
    
    def __init__(self, path, new_im_size, mode):
        super(preprocessing, self).__init__()

        self.path = path
        self.image_path = '../Asi_maskiert/pdfs/'
        self.new_im_size = new_im_size
        self.mode = mode

    def __getitem__(self):
      
        ofile = self.path + '_new_grid.nc'

        #cdo.sellonlatbox(self.lon1, self.lon2, self.lat1, self.lat2, input=ifile, output = ofile)

        ds = xr.load_dataset(ofile, decode_times=False)

        #extract the variables from the file
        if self.mode == 'mask': 
            sst = ds.tho.values
            x = np.isnan(sst)
            n = sst.shape
            for i in range(n[0]):
                for j in range(n[1]):
                    for k in range(n[2]):
                        for l in range(n[3]):
                            if np.isnan(sst[i, j, k, l]) == True:
                                sst[i, j, k, l] = 0
                            else:
                                sst[i, j, k, l] = 1

        elif self.mode == 'image':
            sst = ds.thetao.values
            x = np.isnan(sst)
            n = sst.shape
            sst[x] = 0

        rest = np.zeros((n[0], n[1], self.new_im_size - n[2], n[3]))
        sst = np.concatenate((sst, rest), axis=2)
        n = sst.shape
        rest2 = np.zeros((n[0], n[1], n[2], self.new_im_size - n[3]))
        sst = np.concatenate((sst, rest2), axis=3)
        
        n = sst.shape
        return sst, n


    def plot(self):
        
        sst_new, n = self.__getitem__()
        pixel_plot = plt.figure()
        pixel_plot = plt.imshow(sst_new[0, 0, :, :], vmin = -10, vmax = 40)
        plt.colorbar(pixel_plot)
        #plt.savefig(self.image_path + self.name + '.pdf')
        plt.show()

    def save_data(self):

        sst_new, n = self.__getitem__()

        #create new h5 file with symmetric ssts
        f = h5py.File(self.path + '_newgrid.hdf5', 'w')
        dset1 = f.create_dataset('tos_sym', (n[0], n[1], n[2], n[3]), dtype = 'float32', data = sst_new)
        f.close()



dataset1 = preprocessing('../Asi_maskiert/original_masks/Maske_2020', 128,'mask')
dataset2 = preprocessing('../Asi_maskiert/original_image/Image_r10_11', 128,'image')
#sst, n = dataset1.__getitem__()
#sst2, n2 = dataset2.__getitem__()
#dataset2.plot()
dataset1.save_data()
dataset2.save_data()
#print(sst.shape)
