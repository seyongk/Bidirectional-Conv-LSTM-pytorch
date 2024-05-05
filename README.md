# Bidirectional-Conv-LSTM-PyTorch

## Introduction
This repository contains the implementation of a bidirectional Convolutional LSTM (ConvLSTM) in PyTorch, as described in the paper [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). The ConvLSTM model is particularly useful for spatiotemporal predictions where both spatial and temporal dynamics need to be captured.

## Features
- Implementation of ConvLSTM with bidirectional capability.
- Suitable for tasks like video frame prediction, anomaly detection in videos, and other spatiotemporal sequence prediction tasks.

## Environment Setup
- Python 3.8
- PyTorch 1.13

To set up your environment to run the code, you can follow these steps:
```
bash
conda create --name convlstm python=3.8
conda activate convlstm
pip install torch==1.13 torchvision
```

### Basic Example
```
from model import ConvLSTM
import torch

# Example parameters
input_channels = 3
hidden_channels = 16
kernel_size = (3, 3)
padding = (1, 1)
stride = (1, 1)
bias = True
batch_first = True
bidirectional = True

model = ConvLSTM(
    in_channels=input_channels,
    hidden_channels=hidden_channels,
    kernel_size=kernel_size,
    padding=padding,
    stride=stride,
    bias=bias,
    batch_first=batch_first,
    bidirectional=bidirectional
)

# Example input tensor
batch_size, seq_len, height, width = 1, 10, 64, 64
x = torch.rand(batch_size, seq_len, input_channels, height, width)
output, (hidden_state, cell_state) = model(x)
```
