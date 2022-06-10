# Classification of Melt Layers in Ice Cores

All original data from [Weikusat *et al.*](https://doi.org/10.1594/PANGAEA.925014)
---

Labels and additional meta data are in this repository as \*.tab and \*.xlsx\
Raw images are availabel with two different `integration` options: [None](https://sid.erda.dk/share_redirect/cJWYMvzmm1) and [1](https://sid.erda.dk/share_redirect/gmp24aIh2F)

1. For each set of images the labels need to be corrected as shown here: [Colab Adjust Labels](https://colab.research.google.com/drive/10wXGv9pDnw-I4A4GQkss2vl9PBVHfT0n?usp=sharing)

2. Each image is cropped to the same width, the labels are adjusted and the image is sliced into several smaller images:
[Colab Slice Image](https://colab.research.google.com/drive/1G_R7h1Q72xgkWzrFKPHKqIJHoxq4CnIi?usp=sharing)
The processed slices for `integration=None` are available [here](https://sid.erda.dk/share_redirect/EGXmWRTHi1) and for `integration=1` here **TBD**
with the naming convention:
`{ice_core_number}_{center_pixel_value}_{flip}_{label}.npy`
where `flip` is one of `[o, h, v, hv]` indicating no, horizontal, vertical or horizontal and vertical flip and `label` indicates if the slices contains any melt layer (`1`) or if it doesn't (`0`).

3. A CNN is trained on the `integration=None` slices and achieves an accuracy of 60-70% accuracy on the test set: [Colab CNN](https://colab.research.google.com/drive/1JUhUcL52CUQYvEhcodqmm9RTVGIy6knJ?usp=sharing)

---

**To do:**
- Try more image augmentation
- Try other NN architectures
- Process and slice images with `integration=1`
- Train CNN on new slices
- Combine predictions from the two seperate NN?
