# Find the Assembly Mistakes: Error Segmentation for Industrial Applications

## Check out our [project page](https://timschoonbeek.github.io/error_seg)!

Proper documentation for this repository coming soon!



### Data structure
```
your-data-root-directory
└───Train_set
└───Main_test_set
└───Real_img_test_set
└───Novel_poses_test_set (optional)
└───Novel_parts_test_set (optional)
└───COCO_Images
│   └───unlabeled2017
│   │   000000517440.jpg
│   │   000000517441.jpg
│   │   ...
│   train_img_list.json
│   val_img_list.json
│   test_img_list.json
```

Using the our training set with 200 poses of the assembly object in 5000 states, each directory has the following structure:
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
│   └───sequence199
│   │   │   ...
│   state_list.json
│   state_table.json
│   annotation_defintions.json
│   ...
│   orientation_table.json
```
