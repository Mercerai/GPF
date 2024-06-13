# GPF: Learning Robust Generalizable Radiance Field with Visibility and Feature Augmented Point Representation
<!-- ![issues](https://img.shields.io/github/issues/Mercerai/GPF)
![forks](https://img.shields.io/github/forks/Mercerai/GPF?style=flat&color=orange)
![stars](https://img.shields.io/github/stars/Mercerai/GPF?style=flat&color=red) -->

> [Learning Robust Generalizable Radiance Field with Visibility and Feature Augmented Point Representation](https://iclr.cc/virtual/2024/poster/17836)       
> Jiaxu Wang, Ziyi Zhang, Renjing Xu*    
> ICLR 2024
> 

 ![framework_img](figs/framework.png)

If you found this project useful, please [cite](#citation) us in your paper, this is the greatest support for us.

## Installation
```shell
git clone https://github.com/Mercerai/GPF.git
cd GPF
pip install -r requirements.txt
```
<details>
  <summary> Dependencies </summary>

  - torch==1.7.1
  - numpy==1.19.2
  - CUDA 11.4 or later version

</details>

## Dataset
We reorganize the original datasets in our own format. Here we provide a demonstration of the test set of DTU, which can be downloaded here. After placing the demo data into the data directory, one can directly run the test code as follows. In the data_preprocess dir, we provide the code to reorganize the original datasets into our format. 

## Pretrained models    
We provide the pretrained model which can be applied to the DTU and BlendedMVS datasets. 

## Evaluation for rendering a novel camera trajectory
```shell
python test.py --config ./configs/dtu_config --render_path --num_views 15 --interp_0 14 --interp_1 16 --scan_num 114
```

## Evaluation for rendering a certain camera viewpoint in the test set
```shell
python test.py --scan_num 114 --view_num 36
```
--render_path switch on if camera path is rendered  
--num_views how many frames will be rendered on the path  
--interp_0 and 1 frames between which need to be interpolated  
--scan_num the scan number  
--view_num specific view number  
  
The results will be saved in ./log/dtu_eval/
## Acknowledgements
In this repository, we have used codes or datasets from the following repositories. 
We thank all the authors for sharing great codes or datasets.
- [NeRF-torch](https://github.com/yenchenlin/nerf-pytorch)
- [IBRNet](https://github.com/googleinterns/IBRNet)
- [MVSNet-official](https://github.com/YoYo000/MVSNet) 
- [MVSNeRF](https://github.com/apchenstu/mvsnerf)
- [PixelNeRF](https://github.com/sxyu/pixel-nerf)
- [COLMAP](https://github.com/colmap/colmap)
- [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36)
- [BlendedMVS](https://github.com/YoYo000/BlendedMVS)

## Citation
```
@inproceedings{jiaxu2023learning,
  title={Learning Robust Generalizable Radiance Field with Visibility and Feature Augmented Point Representation},
  author={Jiaxu, WANG and Zhang, Ziyi and Xu, Renjing},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}

```
