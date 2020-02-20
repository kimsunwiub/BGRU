## Bitwise Recurrent Neural Networks: 

About: Incremental approach to binarizing highly sensitive recurrent neural networks. The model is sequentially trained from a real-valued network initially, and re-trained (fine-tuned) with weights quantized at 10% increments until fully binarized. 

This project was supported by Intel Corporation.

Paper: Incremental Binarization On Recurrent Neural Networks For Single-Channel Source Sepration. Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8682595

### Repository structure

#### training_stage_1.py
* First-phase pretraining a real-valued network.

#### training_stage_1.py
* Second-phase incremental binarization process.

### Data Generation
* To be posted soon.

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
