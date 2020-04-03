## Solution of Meta_Learners for 2019 NeurIPS autoDL competition

---
Usage:

This repository is the code folder of the competition. Please:
1. Follow the instructions from [official website](https://github.com/zhengying-liu/autodl_starting_kit_stable) to setup the experimental environment.
2. Download `2nd_General_TinyBERT_4L_312D.zip` and `thin_resnet34.h5` from the releases and put them under `models/` and `speech_util/pretrained_models/` respectively.
3. Run following command to 
get the performance of certain dataset.


```
python run_local_test.py -dataset_dir='PATH_TO_YOUR_DATASET' -code_dir='PATH_TO_THIS_CODE_FOLDER'
```