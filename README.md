# Any-Width Networks

This repository is the official Pytorch implementation of [Any-Width Network](http://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Vu_Any-Width_Networks_CVPRW_2020_paper.html), a real-time adjustable-width CNN architecture that provides maximally fine-grained speed-accuracy trade-off during inference. More information can be found in our [CVPRW 2020 paper](http://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Vu_Any-Width_Networks_CVPRW_2020_paper.html).

<div align="center">
  <img src="https://lh3.googleusercontent.com/pw/ACtC-3f5hZ1Pw180NelSU2M7KXk618GjOzkkpH3U1ihMpkiuqLEipBvaef6aA_Tj9nyuxN42Er5L829Pjo-m6puAi5D1he8iI4VIUxJZ5iRM80c7QAB_vOOqhmkS6qwa5yVzLCrvdBkysG6J2FZEB47Me_hF=w1280-h618-no?authuser=2" width="850" />
</div>

## Quick Start

1. Requirements:
    * torch==1.4.0
    * torchvision==0.5.0
    * numpy==1.18.1
    * PyYAML==5.3
    * matplotlib==3.1.3

      *(other versions may also work, but were not tested)*

2. Training:
    * To train, run: `python train.py cfg:<path-to-yaml-config-file>`
    * For example:
      * `cd <project-root>`
      * `export CUDA_VISIBLE_DEVICES=0`
      * `python train.py cfg:cfg/lenet_cifar10_awn-rs.yml`

3. Testing:
    * To test, uncomment `test_only: True` in the config file used during training
    * Then run: `python train.py cfg:<path-to-yaml-config-file>`

## Acknowledgement

This repository was built on top of Jiahui Yu's [Slimmable Networks](https://github.com/JiahuiYu/slimmable_networks) and Kuang Liu's [CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar). Parts of the code were implemented prior to the official release of Slimmable Networks repo based on their paper and updated afterwards.


## License

This repository is released under the **CC 4.0 Attribution-NonCommercial International License** and should only be used for educational and academic purposes. See [LICENSE](https://github.com/thanhmvu/awn/blob/master/LICENSE) for more details.


## Citation
If you find this repository useful for your own work, please cite our paper:
```BibTeX
@InProceedings{Vu_2020_CVPR_Workshops,
  author = {Vu, Thanh and Eder, Marc and Price, True and Frahm, Jan-Michael},
  title = {Any-Width Networks},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2020}
}
```

