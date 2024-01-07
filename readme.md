# EPILOGUE
This repository contains Python scripts for "EPILOGUE: A Contrastive Learning Method for Automated Gene Function Prediction". 

The learning strategy of EPILOGUE. a) An overview of EPILOGUE: Multiple biological networks are inputted, and the output consists of aggregated representations that are utilized for training SVMs and predicting gene function labels. In the field of gene function prediction, the solid lines represent the input, while the dotted lines represent the output. b) The self-supervised training process to aggregate multiple networks. For each view, edge disturbance is performed to generate a new network, and then original data and augmented data are fed into a distinct encoder to obtain corresponding representations. After that, view-specific representations are aggregated into the consensus embedding, which will be utilized for adjacency matrix reconstruction.By maximizing the agreement between representations of the same node while minimizing the reconstruction error, EPILOGUE dedicates itself to obtaining high-quality representations with discriminative information.
<!-- ![epilogue](flow.png){:height="50%" width="50%"} -->
<div align=center>  
 <img src="flow.png" alt="epilogue" width="70%"/>
 </div>

## Dependencies
EPILOGUE is tested to work under Python 3.10.8.

- pytorch == 2.0.1
- torch-geometric == 2.3.1
- cupy == 12.1.0
- cuml == 23.6.0
- numpy == 1.24.1
- scipy == 1.11.1
- scikit-learn == 1.3.0

## Install

```git
git clone https://github.com/ZxyGed/EPILOGUE.git
```

## Usage
To run EPILOGUE on yeast dataset by default, run the following command from the project directory:
```python 
# generate representations by EPILOGUE
python main.py -d yeast 
# evaluate generated representations
python geneFuncPrediction.py -d yeast -sbi level1 -ebf 'yeast.npy' -msg 'evaluation on yeast dataset' 
```
To see the list of options:
```python
python main.py --help
```