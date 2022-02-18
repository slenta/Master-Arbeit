
from time import time
import numpy as np
import matplotlib.pylab as plt
import cdo
from sympy import N
cdo = cdo.Cdo()
import xarray as xr
import config as cfg
import h5py
import netCDF4


class preprocessing():
    
    def __init__(self, name, new_im_size, lon1, lon2, lat1, lat2, mode):
        super(preprocessing, self).__init__()

        self.path = '../Asi_maskiert/'
        self.image_path = '../Asi_maskiert/pdfs/'
        self.name = name
        self.new_im_size = new_im_size
        self.lon1 = lon1
        self.lon2 = lon2
        self.lat1 = lat1
        self.lat2 = lat2
        self.mode = mode

    def __getitem__(self):
        
        ifile = self.path + self.name + '.nc'
        ofile = self.path + self.name + '_newgrid.nc'

        cdo.sellonlatbox(self.lon1, self.lon2, self.lat1, self.lat2, input=ifile, output = ofile)

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
            #ds = ds.sel(time=slice(24*365.25*46, 24*365.25*63))
            time_var = ds.time
            ds['time'] = netCDF4.num2date(time_var[:],time_var.units)
            ds_monthly = ds.groupby('time.month').mean('time')
            ds = ds.sel(time=slice('2004-01', '2020-10'))

            sst_mean = ds_monthly.thetao.values
            sst = ds.thetao.values

            for i in range(len(sst)):
                sst[i] = sst[i] - sst_mean[i%12]

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
        pixel_plot = plt.imshow(sst_new[0, 0, :, :], vmin = -5, vmax = 5)
        plt.colorbar(pixel_plot)
        #plt.savefig(self.image_path + self.name + '.pdf')
        plt.show()

    def save_data(self):

        sst_new, n = self.__getitem__()

        #create new h5 file with symmetric ssts
        f = h5py.File(self.path + self.name + '_newgrid.hdf5', 'w')
        dset1 = f.create_dataset('tos_sym', (n[0], n[1], n[2], n[3]), dtype = 'float32', data = sst_new)
        f.close()



dataset = preprocessing('original_masks/Maske_2020', 128, -65, -5, 20, 69,'mask')
sst, n = dataset.__getitem__()
#print(sst.shape)
dataset.plot()
dataset.save_data()
#print(sst.shape)
