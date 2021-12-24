# Experiment Programs

## [my\_model](my_model/)
Models for the research

As shown below, the dataset path in "path.py" is hard-coded.
* path.py
```
ILSVRC2012_DATASET_PATH = '/data1/dataset/ilsvrc2012'
```

## [train\_model](train_model/)

To train the models (ResNet and PlainNet)


## [utils](utils/)
Some programs

* To Display Image
* To Conversion Image
* To Load Dataset
* To Calculate of receptive field
* To Track Intermediate Representations
* To Save Settings

## 直下プログラム

** Common Features **


* Use argparse to determine the experimental setup.
* See [batch process](... /batch_process/) for the settings used in the experiments. 
* Be careful about the device you use to store the files, as they tend to be large (over 100GB).

--- 

* [random\_sample.txt](random_sample_val.txt)
  - Subset labels for the dataset used in the analysis (considering the percentage of labels)

* [make\_rf\_datas.py](make_rf_datas.py)
  - Create intermediate files for analysis.                 
  - The main purpose is to cut out the receptive field image and save the gradient information.

* [analyize\_rf\_datas.py](analyize_rf_datas.py)
  - It calculates the Top K receptive fields and the average receptive field. 
  - The details are controlled by arguments.

* [analyize\_optimalinput.py](analyize_optimalinput.py)
  - Visualization by activation maximization method.
  - Adam and L-BFGS are supported.


* [analyize\_preact.py](analysis_preact.py)
  - Analyze the middle receptive field (previous input) of a neuron.
  - The previous input is imaged using a dimensionality reduction technique.

* [analyize\_sparse\_and\_svcca.py](analyize_sparse_and_svcca.py)
  - Graph the count of sparsity of the activity values.
  - Graph the value of SVCCA.
  - The layer to be analyzed is hard-coded.

* [analyize\_numberInClass.py](analyize_numberInClass.py)
  - Analysis of the classes preferred by the neurons

* [train\_transfer\_model.py](train_transfer_model.py)
  - Transfer learning of ResNet.                     
  - transfer-learn specific intermediate layers. 

* [make\_image\_filter.py](make_image_filter.py)
  - Analyze the image in YUV color space.

* [make\_analysis\_meanrfs.py](make_analysis_meanrfs.py)
  - Perform PCA of the receptive field image.

