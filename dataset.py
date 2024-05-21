import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from astropy.io import fits


class MyDataset(Dataset):

    def __init__(self, path_visual_images, sim=True, filters=[0,1,2,3,4], norm="minmax"):

        self.path_visual_images = path_visual_images
        final_size = 600 - 600%32
        self.xmin = (600 - final_size)//2
        self.xmax = self.xmin + final_size
        self.sim = sim
        self.filters = list(filters)
        self.norm = norm


    def __getitem__(self, index):
        
        if self.sim:
            x_ind = fits.getdata(self.path_visual_images[index]).astype(np.single).transpose((1,2,0))

        else:
            x_ind = fits.getdata(self.path_visual_images[index]).astype(np.single)

        y_ind = fits.getdata(self.path_visual_images[index], 'MASK').astype(np.single)
        r_size = fits.getheader(self.path_visual_images[index], 'MASK')['REDGE']
        y_ind = np.where(y_ind<=r_size, 1, 0)

        x_ind = x_ind[self.xmin:self.xmax, self.xmin:self.xmax, self.filters]

        if self.norm=="minmax":
            for ii in range(x_ind.shape[2]):
                x_ind[:,:,ii] = (x_ind[:,:,ii]-x_ind[:,:,ii].min()) / (x_ind[:,:,ii].max()-x_ind[:,:,ii].min())
        
        elif self.norm=="minperc":
            for ii in range(x_ind.shape[2]):
                max_perc = np.percentile(x_ind[:,:,ii],99)
                x_ind[:,:,ii] = (x_ind[:,:,ii]-x_ind[:,:,ii].min()) / (max_perc-x_ind[:,:,ii].min())
                x_ind[:,:,ii] = np.clip(x_ind[:,:,ii], 0, 1)

        elif self.norm=="asinh":  
            for ii in range(x_ind.shape[2]):
                max_perc = np.percentile(x_ind[:,:,ii],99)
                x_ind[:,:,ii] = (x_ind[:,:,ii]-x_ind[:,:,ii].min()) / (max_perc-x_ind[:,:,ii].min())
                x_ind[:,:,ii] = np.arcsinh(x_ind[:,:,ii]/1.1752011936438014)
                x_ind[:,:,ii] = np.clip(x_ind[:,:,ii], 0, 1)


        x_ind = np.where(np.isfinite(x_ind), x_ind, 0)

        y_ind = y_ind[self.xmin:self.xmax, self.xmin:self.xmax]

        x_ind = np.moveaxis(x_ind, -1, 0)
        x_ind = torch.from_numpy(x_ind.copy()).float()
        y_ind = torch.from_numpy(y_ind.copy()).long()
        

        return x_ind, y_ind

    def __len__(self):

        return len(self.path_visual_images)


class MyDatasetNormalRotationAndFlip(Dataset):

    def __init__(self, path_visual_images, sim=True, filters=[0,1,2,3,4], norm="minmax"):

        self.path_visual_images = path_visual_images
        final_size = 600 - 600%32
        self.xmin = (600 - final_size)//2
        self.xmax = self.xmin + final_size
        self.sim = sim
        self.filters = list(filters)
        self.norm = norm


    def __getitem__(self, index):

        if self.sim:
            x_ind = fits.getdata(self.path_visual_images[index]).astype(np.single).transpose((1,2,0))
        
        else:
            x_ind = fits.getdata(self.path_visual_images[index]).astype(np.single)

        y_ind = fits.getdata(self.path_visual_images[index], 'MASK').astype(np.single)
        r_size = fits.getheader(self.path_visual_images[index], 'MASK')['REDGE']
        y_ind = np.where(y_ind<=r_size, 1, 0)


        x_ind = x_ind[self.xmin:self.xmax, self.xmin:self.xmax, self.filters]
        
        if self.norm=="minmax":
            for ii in range(x_ind.shape[2]):
                x_ind[:,:,ii] = (x_ind[:,:,ii]-x_ind[:,:,ii].min()) / (x_ind[:,:,ii].max()-x_ind[:,:,ii].min())
        
        elif self.norm=="minperc":
            for ii in range(x_ind.shape[2]):
                max_perc = np.percentile(x_ind[:,:,ii],99)
                x_ind[:,:,ii] = (x_ind[:,:,ii]-x_ind[:,:,ii].min()) / (max_perc-x_ind[:,:,ii].min())
                x_ind[:,:,ii] = np.clip(x_ind[:,:,ii], 0, 1)

        elif self.norm=="asinh":  
            for ii in range(x_ind.shape[2]):
                max_perc = np.percentile(x_ind[:,:,ii],99)
                x_ind[:,:,ii] = (x_ind[:,:,ii]-x_ind[:,:,ii].min()) / (max_perc-x_ind[:,:,ii].min())
                x_ind[:,:,ii] = np.arcsinh(x_ind[:,:,ii]/1.1752011936438014)
                x_ind[:,:,ii] = np.clip(x_ind[:,:,ii], 0, 1)


        x_ind = np.where(np.isfinite(x_ind), x_ind, 0)

        y_ind = y_ind[self.xmin:self.xmax, self.xmin:self.xmax]

        angle = np.random.choice(4)
        flip = np.random.choice(['0', 'v', 'h'])

        x_ind = np.rot90(x_ind, angle)
        y_ind = np.rot90(y_ind, angle)


        if flip == 'v':

            x_ind = np.flip(x_ind, 0)
            y_ind = np.flip(y_ind, 0)

        elif flip == 'h':

            x_ind = np.flip(x_ind, 1)
            y_ind = np.flip(y_ind, 1)

        x_ind = np.moveaxis(x_ind, -1, 0)
        #print(x_ind.shape)
        
        x_ind = torch.from_numpy(x_ind.copy()).float()
        y_ind = torch.from_numpy(y_ind.copy()).long()
        
        #noise_x_ind = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_ind.shape))).float()
        #x_ind = torch.from_numpy(x_ind).float() + noise_x_ind


        return x_ind, y_ind

    def __len__(self):

        return len(self.path_visual_images)




