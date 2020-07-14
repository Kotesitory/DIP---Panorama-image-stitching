# Panorama Stitching

This is a small repository for a small Python project dedicated to panorama image stitching. There are tree algorithms implemented including: Planar stitching, Cylindrical stitching and Hybrid stitching (a combination of both)


## Requirements
- Python 3
- OpenCV 3.4.2
- NumPy 1.16.5

## Usage

All code is packed in a single Python file [*main.py*](https://github.com/Kotesitory/DIP---Panorama-image-stitching/blob/master/main.py). The different stitching techniques are contained in their own function.
```
planar_stitch()
cylindrical_stitch()
cylindrical_planar_stitch()
```
The rest are helper functions. The current implementation expect a path to a *txt* file as a command line argument . The *txt* file must contain the paths to the images. Each image path is in a new row (\r\n separated) and the images are ordered from left to right (as they appear in the panorama). Example:
```
image1.jpg
image2.jpg
image3.jpg
...
```
Use the functions as you wish, but in this version the code calls each one with the same images and also calls the official OpenCV Stitcher implementation for comparison. The user is notified if an algorithm fails and the images are displayed and saved at the end. The images are stored in the current working directory or a directory specified as an optional second command line argument. The images are named: *planar_pano.jpg, cylindrical_pano.jpg, hybrid_pano.jpg* and *official_pano.jpg* respectively.

## Installation
Installation very simple. Only clone the repository and execute the *main. py* file with a Python 3 interpreter. Provide the images in the format specified above and everything should work.

## References
- [1] Kushal Vyas - [Image stitching, a simple tutorial](https://kushalvyas.github.io/stitching.html)
- [2] Roy Shilkrot - [Cylindrical image warping for panorama stitchingl](https://www.morethantechnical.com/2018/10/30/cylindrical-image-warping-for-panorama-stitching/)
- [3] Saurabh Kemekar, Arihant Gaur, Pranav Patil, Danish Gadas - [Image stitching in a planar and cylindrical coordinate system](https://github.com/saurabhkemekar/PANORAMA))