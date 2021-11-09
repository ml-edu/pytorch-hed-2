# Pytorch Holistically-Nested Edge Detection (HED)

[![CodeFactor](https://www.codefactor.io/repository/github/davidelanz/pytorch-hed/badge)](https://www.codefactor.io/repository/github/davidelanz/pytorch-hed)
[![Documentation Status](https://readthedocs.org/projects/pytorch-hed/badge/?version=latest)](https://pytorch-hed.readthedocs.io)
[![travisCI](https://travis-ci.com/Davidelanz/pytorch-hed.svg?branch=master)](https://travis-ci.com/github/Davidelanz/pytorch-hed)
[![codecov](https://codecov.io/gh/Davidelanz/pytorch-hed/branch/master/graph/badge.svg?token=MAAZAVSORH)](https://codecov.io/gh/Davidelanz/pytorch-hed)
[![Pypi](https://img.shields.io/badge/Pypi-pytorch–hed-yellow)](https://pypi.org/project/pytorch-hed/)

- Website: [davidelanz.github.io/pytorch-hed](https://davidelanz.github.io/pytorch-hed/)
- Documentation: [pytorch-hed.readthedocs.io](https://pytorch-hed.readthedocs.io/)

> This is a reimplementation in the form of a python package of Holistically-Nested Edge Detection [[1]](#references) using PyTorch based on the previous pytorch implementation by [sniklaus](https://github.com/sniklaus) [[2]](#references). If you would like to use of this work, please cite the paper accordingly. Also, make sure to adhere to the licensing terms of the authors. Moreover, if you will be making use of this particular implementation[[3]](#references), please acknowledge it.

<a href="https://arxiv.org/abs/1504.06375" rel="Paper"><img src="http://www.arxiv-sanity.com/static/thumbs/1504.06375v2.pdf.jpg" alt="Paper" width="100%"></a>

|   |GitHub|Ref|
|---|---|---|
|Original version based on Caffe | https://github.com/s9xie/hed | [[1]](#references) |
|Another reimplementation based on Caffe | https://github.com/zeakey/hed |
|Original reimplementation based on PyTorch | https://github.com/sniklaus/pytorch-hed | [[2]](#references)|

## Usage

First, you have to install the package (**stable**) with 
```
pip install pytorch-hed
```
or, for the current (**unstable**) version 
```
pip install git+https://github.com/Davidelanz/pytorch-hed.git
```

Usage:

``` python
import torchHED
   
# process a single image file 
torchHED.process_file("./images/sample.png", "./images/sample_processed.png")

# process all images in a folder
torchHED.process_folder("./input_folder", "./output_folder")

# process a PIL.Image loaded in memory and return a new PIL.Image
# img = PIL.Image.open("./images/sample.png")
img_hed = torchHED.process_img(img)
```

## Results

<table>
<thead>
<tr>
<th>Input</th>
<th>Original Caffe Implementation <a href="#references">[1]</a></th>
<th>pytorch-hed <a href="#references">[3]</a></th>
</tr>
</thead>
<tbody>
<tr>
<td><a target="_blank" rel="noopener noreferrer" href="https://github.com/Davidelanz/pytorch-hed/blob/master/images/sample.png?raw=true"><img src="https://github.com/Davidelanz/pytorch-hed/raw/master/images/sample.png?raw=true" alt="sample" style="max-width:100%;"></a></td>
<td><a target="_blank" rel="noopener noreferrer" href="https://github.com/Davidelanz/pytorch-hed/blob/master/images/official_caffe.png?raw=true"><img src="https://github.com/Davidelanz/pytorch-hed/raw/master/images/official_caffe.png?raw=true" alt="sample" style="max-width:100%;"></a></td>
<td><a target="_blank" rel="noopener noreferrer" href="https://github.com/Davidelanz/pytorch-hed/blob/master/images/torchHED.png?raw=true"><img src="https://github.com/Davidelanz/pytorch-hed/raw/master/images/torchHED.png?raw=true" alt="sample" style="max-width:100%;"></a></td>
</tr>
</tbody>
</table>

## References

```
[1]  @inproceedings{Xie_ICCV_2015,
         author = {Saining Xie and Zhuowen Tu},
         title = {Holistically-Nested Edge Detection},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2015}
     }
```

```
[2]  @misc{pytorch-hed,
         author = {Simon Niklaus},
         title = {A Reimplementation of {HED} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-hed}}
    }
```


```
[3]  @misc{pytorch-hed-2,
         author = {Davide Lanza},
         title = {The {pytorch-hed} Python Package},
         year = {2020},
         howpublished = {\url{https://github.com/Davidelanz/pytorch-hed}}
    }
```
