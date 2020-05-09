# TextMountain: Accurate Scene Text Detection via Instance Segmentation

## Introduction
This is the code of TextMountain. It is modified based on [semantic-segmentation-pytorch](https://github.com/open-mmlab/mmdetection).

## Requirements
* Python 3.6.1
* opencv-python 3.4
* torch 0.4.0
* Install other required packages according to the error message

## Compiling grouping code
Our grouping code is based on [RoIAlign.pytorch](https://github.com/longcw/RoIAlign.pytorch)
```
cd groupSearch
bash make.sh
cd ..
cd groupmeanScore/
bash make.sh
```

## Training
We package labels in roidb_mlt.pik and roidb_ctw.pik. If you want to train with your own dataset, you can refer to our packaging format and make a new annotation file.
```
python train_ctw.py
python train_mlt.py
```

## Evaluation
```
python eval_ctw.py
python eval_mlt.py
```
## Reference

If you use this codebase or models in your research, please consider cite .

```
@article{zhu2018textmountain,
  title={Textmountain: Accurate scene text detection via instance segmentation},
  author={Zhu, Yixing and Du, Jun},
  journal={arXiv preprint arXiv:1811.12786},
  year={2018}
}

@article{zhou2018semantic,
  title={Semantic understanding of scenes through the ade20k dataset},
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  journal={International Journal on Computer Vision},
  year={2018}
}

@inproceedings{zhou2017scene,
    title={Scene Parsing through ADE20K Dataset},
    author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2017}
}
```

