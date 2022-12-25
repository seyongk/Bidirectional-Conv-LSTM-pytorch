# Bidirectional-Conv-LSTM-pytorch
Implementation of bidirectional Convolutional LSTM in PyTorch.

## Environment
Python=3.8  
Pytorch=1.13


## Citation  
```
@inproceedings{10.5555/2969239.2969329,
author = {Shi, Xingjian and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-kin and Woo, Wang-chun},
title = {Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting},
year = {2015},
publisher = {MIT Press},
address = {Cambridge, MA, USA},
abstract = {The goal of precipitation nowcasting is to predict the future rainfall intensity in a local region over a relatively short period of time. Very few previous studies have examined this crucial and challenging weather forecasting problem from the machine learning perspective. In this paper, we formulate precipitation nowcasting as a spatiotemporal sequence forecasting problem in which both the input and the prediction target are spatiotemporal sequences. By extending the fully connected LSTM (FC-LSTM) to have convolutional structures in both the input-to-state and state-to-state transitions, we propose the convolutional LSTM (ConvLSTM) and use it to build an end-to-end trainable model for the precipitation nowcasting problem. Experiments show that our ConvLSTM network captures spatiotemporal correlations better and consistently outperforms FC-LSTM and the state-of-the-art operational ROVER algorithm for precipitation nowcasting.},
booktitle = {Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1},
pages = {802–810},
numpages = {9},
location = {Montreal, Canada},
series = {NIPS'15}
}
```
