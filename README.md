## Bitwise Recurrent Neural Networks: 

About: 

Acknowledgements:

### Repository structure

#### training_stage_1.py
*

### Data Generation
* 

### Training stage 1
* To pretrain the real-valued network, 
```
python training_stage_1.py 5 5 3 0.0001 0.4 0.9 50 1
```

### Training stage 2
* To perform incremental binarization,
```
python training_stage_2.py 5 5 9e-5 0.1 Phase_1_Ep500
```

### Datasets used in this repository
* TIMIT (https://catalog.ldc.upenn.edu/LDC93S1)
* Duan (http://www2.ece.rochester.edu/~zduan/data/noise/)
