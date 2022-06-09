# Classification of Melt Layers in Ice Cores

Labels and additional meta data are in this repository as \*.tab and \*.xlsx\
Find raw images with same camera setting:
> https://sid.erda.dk/share_redirect/cJWYMvzmm1


The labels can't be correctly located so we adjust the labels as shown here:
> https://colab.research.google.com/drive/10wXGv9pDnw-I4A4GQkss2vl9PBVHfT0n?usp=sharing


After adjusting the labels and cropping the images to the same width the image is cut into several slices:
> https://colab.research.google.com/drive/1G_R7h1Q72xgkWzrFKPHKqIJHoxq4CnIi?usp=sharing

The processed slices are stored here:
> https://sid.erda.dk/share_redirect/EGXmWRTHi1

with the naming convention:
> *{ice_core_number}_{center_pixel_value}_{flip}_{label}.npy*

where flip is one of [o, h, v, hv] indicating no, horizontal, vertical or horizontal and vertical flip.

A CNN is trained on those slices and gets an accuracy of 60-70% accuracy on the test set:
> https://colab.research.google.com/drive/1JUhUcL52CUQYvEhcodqmm9RTVGIy6knJ?usp=sharing
