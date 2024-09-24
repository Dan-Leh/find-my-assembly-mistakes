# Find the Assembly Mistakes: Error Segmentation for Industrial Applications

## Check out our [project page](https://timschoonbeek.github.io/error_seg)!


### How to train the model
Before training StateDiffNet, edit the `scripts/train.sh` file by putting the relevant file paths to the data directories, as well as where you want your model to save checkpoints, and other results (eg. --output_root for where to save logging and qualitative results during training). Additionally, you can set flags for all of the hyperparameters, or edit them in the config file (the parameters set in the bash file overwrite the parameters in the config file). All configuration parameters for training can be found in the `config_args.py` file, along with detailed explanations of how each parameter influences the training. An example configuration file can be found under `config_files/example_train`, containing recommended hyperparameter values. Once you have chosen all your config parameters, you can run training from the repository root directory by calling:
```sh scripts/train.sh```

### How to test a model
Before testing a model, open the `scripts/test.sh` file. Here, you can choose all the synthetic test sets that you want to test your model on. These incldue "Main_test_set", "Novel_poses_test_set", "Novel_parts_test_set", "Random_background". These are the options for the `--test_set_name` parameter. "Random_background" contains the same images as the main test set, but where the background is randomized using images from MS COCO. Additionally, indicate the path of the config file used to train the model, and choose the same experiment name as the trained model. Finally, run testing by calling
```sh scripts/test.sh```

### Real-world test set
To test a model on our real-world test pairs, simply indicate the path to the config file of the model in `scripts/test_real_imgs` and run:
```sh scripts/test_real_imgs.sh```

### Data structure
We recommend to organize all of the training/test sets in the same directory, (whose path can be specified as one argument in the config file) as follows:
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
│   resnet_pretrained_weights
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
