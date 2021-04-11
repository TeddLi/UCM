# UCM

This repository contains the source code and data for the ICASSP 2021 paper:

HAVE YOU MADE A DECISION? WHERE? A PILOT STUDY ON INTERPRETABILITY OFPOLARITY ANALYSIS BASED ON ADVISING PROBLEM

## Dependencies

Python 3.6
Tensorflow 1.14.0



## Dataset statistics

<img src="Image/datastatistics.png">


## Results

<img src="Image/performance.png">



## Test the trained model

A checkpoint is saved in scripts/runs/restore/
The test can be conducted directly

```
cd scripts
bash eval.sh
```

## Get the results of all evaluation matrics
```
python evaluation_metric.py
```


## Train a new model
```
cd scripts
bash train_ucm.sh
```
