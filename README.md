# Find the Assembly Mistakes: Error Segmentation for Industrial Applications

## Check out our [project page](https://timschoonbeek.github.io/error_seg)!

Proper documentation for this repository coming soon!





### Data structure
We recommend to organize all of the training/test sets in the same directory, as follows:
```
your-data-root-directory
└───Train_set
└───Main_test_set
└───Real_img_test_set
└───Novel_poses_test_set (optional)
└───Novel_parts_test_set (optional)
└───COCO_Images (optional)
│   └───unlabeled2017
│   │   000000517440.jpg
│   │   000000517441.jpg
│   │   ...
│   train_img_list.json
│   val_img_list.json
│   test_img_list.json
```

Using the our training set as an example, it contains 200 poses of the assembly object in 5000 states. In the file structure, this corresponds to 200 folders, each containing 5000 images, segmentation masks, and json files containing the annotation data. The naming is as follows, with each sequence number corresponding to one pose, and each step corresponding to a unique state:
```
└───Train_set
│   └───sequence0000
│   │   │   step0000.camera.instance segmentation.png
│   │   │   step0000.camera.png
│   │   │   step0000.frame_data.json
│   │   │   ...
│   │   │   step4999.camera.instance segmentation.png
│   │   │   step4999.camera.png
│   │   │   step4999.frame_data.json
│   └───sequence0001
│   │   │   ...
│   │   ...
│   └───sequence0199
│   │   │   ...
│   state_list.json
│   state_table.json
│   annotation_defintions.json
│   ...
│   orientation_table.json
```


### Acknowledgments
Much of the code in this repositor and some of the ideas in our paper were taken from and inspired by the following sources. We thank the authors for making their work open-source.
- [The Change You Want to See](https://github.com/ragavsachdeva/The-Change-You-Want-to-See)
- [Remote Sensing Image Change Detection with Transformers](https://github.com/justchenhao/BIT_CD)
- [LoFTR: Detector-Free Local Feature Matching with Transformers](https://github.com/zju3dv/LoFTR/tree/master)
- [PyTorch Segmentation Models](https://github.com/jlcsilva/segmentation_models.pytorch)
