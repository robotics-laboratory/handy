# Detection

This is module for detection of table tennis balls in images. It utilizes SSD model with series of custom backbones. The code for training, inferencing and measuring the speed of the model is available here.

## Installation

To install the dependencies, run the following command:

``` bash
pip install -r requirements.txt
```

## Training

### Data preparation

`datasets.py` contains the code for creating the dataset.

Classes:

* DetectionDataset: A PyTorch Dataset class for the detection task.
* DetectionDataModule: A PyTorch Lightning DataModule class for the detection task.

For checking the dataset, run the following command:

``` bash
python detection/tdatasets.py --image_dir path/to/images --annot_file path/to/annotations --width 500 --height 500
```

Annotaion should be in the following format:

``` json
{"image_name.png": 
    {"xmin": ... , "ymin": ... , "xmax": ... , "ymax": ... },
    ...
}
```

It can be created from segmentatin masks using `generate_bounding_boxes` function from `datasets.py`.

## Train

To train the model, run the following command:

```bash
python detection/train.py --data_dir {path_to_images_directory} --annot_file {path_to_annotations_file} --width {image_width} --height {image_height} --backbone {backbone_model} --batch_size {batch_size} --epochs {number_of_epochs}
```

Where:

* {path_to_images_directory} is the path to the directory containing the images.
* {path_to_annotations_file} is the path to the file containing the bounding box annotations.
* {image_width} is the width to which the images will be resized.
* {image_height} is the height to which the images will be resized.
* {backbone_model} is the backbone model to be used for the detection task from the following options:
  * resnet34
  * vovnet39
  * mobilenet_v2
  * mobilenet_v3
  * mobilenet_v3_large

* {batch_size} is the number of samples per gradient update.
* {number_of_epochs} is the number of epochs to train the model.

## Inference

To run inference on the trained model, run the following command:

```bash
python detection/inference.py --data_dir {path_to_images} --result_dir {path_to_results} --backbone {backbone_model} --checkpoint {path_to_checkpoint} --threshold {threshold} --size {image_size}
```

## Speed measurement

To measure the speed of the model, run the following command:

```bash
python detection/speed_test.py --backbone {backbone_model} --checkpoint {path_to_checkpoint} --size {image_size} --img_path {path_to_image}
```
