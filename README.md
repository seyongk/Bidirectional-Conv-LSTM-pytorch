# Bidirectional-Conv-LSTM-PyTorch

## Introduction
This repository contains the implementation of a bidirectional Convolutional LSTM (ConvLSTM) in PyTorch, as described in the paper [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). The ConvLSTM model is particularly useful for spatiotemporal predictions where both spatial and temporal dynamics need to be captured.

## Features
- Implements bidirectional ConvLSTM, allowing the model to capture both forward and backward temporal dependencies
- Supports batch processing and variable sequence lengths
- Provides configurable parameters for input channels, hidden channels, kernel size, padding, stride, and bias
- Includes a custom `ConvGate` module for efficient computation of gate activations
- Compatible with PyTorch's `nn.Module` and can be easily integrated into existing PyTorch models


## Environment Setup
- Python 3.8
- PyTorch 1.13

To set up your environment to run the code, you can follow these steps:
```
conda create --name convlstm python=3.8
conda activate convlstm
pip install torch==1.13 torchvision
```

### Basic Example
```
from bidirectional_conv_lstm import ConvLSTM

# Create a ConvLSTM instance
conv_lstm = ConvLSTM(
    in_channels=1,
    hidden_channels=64,
    kernel_size=3,
    padding=1,
    stride=1,
    bias=True,
    batch_first=True,
    bidirectional=True
)

# Forward pass
input_tensor = torch.randn(batch_size, seq_len, in_channels, height, width)
output, (hidden_state, cell_state) = conv_lstm(input_tensor)
```

### References

Shi, X., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in neural information processing systems (pp. 802-810). [arXiv:1506.04214]

### Contributing
Contributions to this repository are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License.
