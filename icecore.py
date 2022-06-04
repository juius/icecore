import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread

from skimage.measure import block_reduce
from pathlib import Path

d = Path(__file__).resolve().parent

class IceCore:
    def __init__(self, img_file):
        self.img_file = Path(img_file)
        self.image_raw = imread(self.img_file)
        self.set_parameters()
        
    def set_parameters(
            self,
            n_center_pixels=1000,
            first_n_pixels=700,
            last_n_pixels=700,
            downsample=10,
            normalize=False,
            pixels_per_cm=186):
        self.process_params = {
            'n_center_pixels': n_center_pixels,
            'first_n_pixels': first_n_pixels,
            'last_n_pixels': last_n_pixels,
            'downsample': downsample,
            'normalize': normalize
        }
        self.pixels_per_cm = pixels_per_cm

    def _detect_edge(self, from_where):
        assert from_where in ['top', 'bot', 'bottom'],\
            f'Choose either top or bottom'
        if from_where == 'top':
            modimg = np.copy(
                self.image_raw[:self.process_params['first_n_pixels'], :])
        elif from_where in ['bot', 'bottom']:
            modimg = np.copy(
                self.image_raw[-self.process_params['last_n_pixels']:, :])
        # reduce resolution
        smaller = block_reduce(
            modimg,
            (self.process_params['downsample'],
             self.process_params['downsample']),
            np.mean)
        # horizontal edge detection with Sobel kernel
        sobely = cv2.Sobel(smaller, cv2.CV_64F, 0, 1, ksize=5)
        # find min and max along rows
        min = sobely.mean(axis=1).argmin()
        max = sobely.mean(axis=1).argmax()
        edge_smaller = np.mean([min, max])
        # convert smaller back to real position in img
        edge = edge_smaller * self.process_params['downsample']
        if from_where in ['bot', 'bottom']:
            edge += self.image_raw.shape[0] - \
                self.process_params['last_n_pixels']
        return int(edge)

    def preprocess_image(self):
        # cut off left and right of image
        height, width = self.image_raw.shape
        n_cut = width - self.process_params['n_center_pixels']
        cut_left = int(np.floor(n_cut / 2))
        cut_right = int(np.ceil(n_cut / 2))
        center_img = self.image_raw[:, cut_left:width - cut_right]
        # cut off top and bottom of image
        start = self._detect_edge(from_where='top')
        end = self._detect_edge(from_where='bottom')
        self.image = center_img[start:end, :]
        # normalize picture
        if self.process_params['normalize']:
            self.image = cv2.normalize(
                self.image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F)

    def locate_image(
            self,
            meta_data_file=d/'EGRIP_visual_strat.tab'):
        meta_df = pd.read_csv(
            meta_data_file,
            sep='\t',
            skiprows=24,
            header=0)
        # only keep info on pictures from the right camera
        mask = list(np.array([len(name.split('_'))
                    for name in meta_df['File name']]) == 1)
        meta_df = meta_df[mask]
        meta_data = meta_df[meta_df['File name']
                            == self.img_file.name]
        self.top = meta_data['Depth top [m]'].values[0]
        theo_bottom = meta_data['Depth bot [m]'].values[0]
        self.length = self.image.shape[0] / (self.pixels_per_cm * 100)
        self.bottom = self.top + self.length
        assert np.abs(self.bottom - theo_bottom) < 0.05 * self.bottom, \
            f"Predicted and theoretical bottom of IceCore differ by more than 5%:\n\tPred: {self.bottom:.02f}\n\tTheo: {theo_bottom:.02f}"

    def load_layers(
            self,
            label_file=d/'Westhoff_et_al_Melt_Events_and_other_Bubble-free_Features_in_the_EastGRIP_Ice_Core.xlsx'):
        label_df = pd.read_excel(label_file, sheet_name=1)
        df_molten = label_df[label_df['type of feature']
                             == 'meltLayer']
        layers_df = df_molten[(df_molten['depth from'] >= self.top) & (
            df_molten['depth to'] <= self.bottom)]
        self.layers = []
        self.layers_pxl = []
        for _, row in layers_df.iterrows():
            self.layers.append((row['depth from'], row['depth to']))
            self.layers_pxl.append((self.meter2pixel(row['depth from']), self.meter2pixel(row['depth to'])))
        if len(self.layers) > 0:
            self.has_melt_layer = True
        else:
            self.has_melt_layer = False

    def adjust_layers(self, shift):
        assert len(self.layers) > 0,\
            f'No melt layers in this photo'
        self.layers_pxl_adj = []
        for l in self.layers_pxl:
            self.layers_pxl_adj.append(tuple([m+shift for m in l]))
        self.layers_adj = []
        for adj_l in self.layers_pxl_adj:
            self.layers_adj.append(tuple([self.pixel2meter(p) for p in adj_l]))

    def autofix_layers(self):
        intensities = np.mean(self.image,axis=1) * np.median(self.image,axis=1)
        explore = 200 # looks for minimum of cost function 200 pixels to left and right
        extra = 20 # considers 20 pixels to left and right from layer as well
        layers = self.layers
        layers_in_pixels = []
        for l in layers:
            layer_position = [int(self.meter2pixel(m)) for m in l]
            layers_in_pixels.append(np.arange(layer_position[0]-extra, layer_position[1]+extra))
        layers_mask = np.hstack(layers_in_pixels)
        
        shift_range = np.arange(-explore,explore)
        costs = []
        for shift in shift_range:
            costs.append(np.sum(intensities[layers_mask+shift]))
        self.cost4shift = (shift_range, np.asarray(costs))
        self.shift = shift_range[np.argmin(costs)]        
        self.adjust_layers(self.shift)
        
    def plot_shift_cost(self):
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(*self.cost4shift)
        ax.scatter(self.shift, np.min(self.cost4shift[1]), color='crimson', s=100)
        ax.set_ylabel('Cost Function')
        _ = ax.set_xlabel('Shift in Pixels')
        print(f'The best shift for {self.img_file.name} is {self.shift} pixels')

    def pixel2meter(self, pixel):
        return pixel / (self.pixels_per_cm * 100) + self.top

    def meter2pixel(self, meter):
        pixel = np.round((meter - self.top)
                         * (self.pixels_per_cm * 100)).astype(int)
        assert np.all((pixel <= self.image.shape[0]) & (pixel >= 0)),\
            f"{meter} m is not shown in this image from {self.top} m to {self.bottom} m"
        return pixel

    def process_image(self):
        self.preprocess_image()
        self.locate_image()
        self.load_layers()

    def plot_image(
        self, zoomed=(
            0, 1000), inmeters=False, figsize=(
            12, 12)):
        fig, ax = plt.subplots(figsize=figsize)
        if inmeters:
            zoomed_pixels = [self.meter2pixel(m) for m in zoomed]
        else:
            zoomed_pixels = zoomed
        zommed_img = self.image[zoomed_pixels[0]:zoomed_pixels[1], :]
        ax.imshow(zommed_img)
        zommed_meters = [self.pixel2meter(p) for p in zoomed_pixels]
        meters = np.unique(np.round(np.linspace(*zommed_meters,100),decimals=2))
        meters2pixels_zoomed = [self.meter2pixel(m)-zoomed_pixels[0] for m in meters]
        ax.set_yticks(meters2pixels_zoomed)
        ax.set_yticklabels(meters)
        ax.set_ylabel('Depth in Meters')
        ax.set_title(self.img_file.name)

    def plot_edge_detection(self):
        titles = ['Raw Image', 'Downsampled', 'Edge Detection']
        fig, axes = plt.subplots(
            ncols=3, nrows=2, figsize=(
                12, 4))
        for i, where in enumerate(['top', 'bottom']):
            if where == 'top':
                img = self.image_raw[:self.process_params['first_n_pixels'], :]
            elif where == 'bottom':
                img = self.image_raw[-self.process_params['last_n_pixels']:, :]
            smaller = block_reduce(
                img,
                (self.process_params['downsample'],
                 self.process_params['downsample']),
                np.mean)
            sobely = cv2.Sobel(smaller, cv2.CV_64F, 0, 1, ksize=5)
            edge = self._detect_edge(where)
            if where == 'bottom':
                edge -= self.image_raw.shape[0] - \
                    self.process_params['last_n_pixels']
            edges = [
                edge,
                edge /
                self.process_params['downsample'],
                edge /
                self.process_params['downsample']]
            axs = axes[i]
            for j, dat in enumerate([img, smaller, sobely]):
                axs[j].imshow(dat)
                axs[j].scatter(int(dat.shape[1] / 2),
                               edges[j], color='red', alpha=0.5)
                if i < 1:
                    axs[j].set_title(titles[j])
        fig.tight_layout

    def plot_layers(self, pad=100, adjusted=True):
        if adjusted:
            layers = self.layers_adj
        else:
            layers = self.layers
        n_layers = len(layers)
        assert n_layers > 0,\
            f'No melt layers in this photo'
        fig, axs = plt.subplots(
            nrows=n_layers, figsize=(
                12, n_layers * 5), sharex=False)
        if n_layers == 1:
            axs = [axs]
        for i, label in enumerate(layers):
            pixel_positions = [self.meter2pixel(l) for l in label]
            cut_image = self.image[pixel_positions[0] -
                                   pad:pixel_positions[1] + pad, :]
            axs[i].imshow(cut_image, interpolation='none')
            org_lim = axs[i].get_xlim()
            axs[i].hlines([pad, cut_image.shape[0] - pad],
                          *org_lim, color='red')
            axs[i].set_xlim(org_lim)

