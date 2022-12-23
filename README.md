## M-CRBMs
&ensp; This repo implements the modified conditional restricted Boltzmann machines (M-CRBMs) model in paper *Modified Conditional Restricted Boltzmann Machines for Query
Recommendation in Digital Archives*. The paper is in submitting progress.
<br/><br/>

## Data  
   &ensp; Due to the huge size of the original training/test dataset, we only provide a small part of them in this repo.   
   &ensp; To protect user information, we do not provide real search keywords, but only the search keyword IDs in training/test dataset.
<br/>

   ### **Data Files**
   `dataset_train_<category>.npy`
   `dataset_test_<category>.npy`
<br/><br/>

## How to Use
### **Install Environemt**
&ensp; This repo is tested on pytorch version 1.4.0 (CPU ONLY) on macOS, please ensure that correct pytorch version with CUDA version is installed on your computer.  
&ensp; Install the same pytorch version and other dependencies as the developer by directly run this conda command:
```
conda env create -f environment.yml
```
<br/>

### **Train & Test M-CRBMs Model**

```
python train_crbm.py
```
Trained models are saved to `model.pth`, and all the metrics are saved to `metrics.csv`
<br/>

### **Set Parameter Values**
&ensp; If you want to set training/test parameter values manually, set them in `./crbm/config.py`. The descriptions of the main parameters are listed below.
|Parameter Name|Description|
|---|---|
|`HIDDEN_UNITS`|Number of nodes in the hidden layer.|
|`BATCH_SIZE`|Batch size for both training data and test data.|
|`EPOCHS`|Training epoches.|
|`LEARNING_RATE`|Learning rate of the Adam optimizer (ignore this if use different optimizer).|
|`WEIGHT_DECAY`|Weight decay of the Adam optimizer (ignore this if use different optimizer).|
|`CD_K`|Number of steps of Gibbs sampling in contrastive divergence, default is 1.|
|`OPTIM`|Set optimizer, 'adam' or 'rms'.|
|`CATEGORY`|What categories will be used, this will be shown in trained model's file name.|
<br/>

