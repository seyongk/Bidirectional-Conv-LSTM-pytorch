from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor, nn, relu, sigmoid, softmax, tanh


class ConvGate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        padding: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int],
        bias: bool,
    ):
        super(ConvGate, self).__init__()
        self.conv_x = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.bn2d = nn.BatchNorm2d(hidden_channels * 4)

    def forward(self, x, hidden_state):
        gated = self.conv_x(x) + self.conv_h(hidden_state)
        return self.bn2d(gated)


class ConvLSTMCell(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, kernel_size, padding, stride, bias
    ):
        super().__init__()
        self.gates = nn.ModuleList(
            [ConvGate(in_channels, hidden_channels, kernel_size, padding, stride, bias)]
        )

    def forward(
        self, x: Tensor, hidden_state: Tensor, cell_state: Tensor
    ) -> Tuple[Tensor, Tensor]:
        gated = self.gates[0](x, hidden_state)
        i_gated, f_gated, c_gated, o_gated = gated.chunk(4, dim=1)

        i_gated = sigmoid(i_gated)
        f_gated = sigmoid(f_gated)
        o_gated = sigmoid(o_gated)

        cell_state = f_gated.mul(cell_state) + i_gated.mul(tanh(c_gated))
        hidden_state = o_gated.mul(tanh(cell_state))

        return hidden_state, cell_state


class ConvLSTM(nn.Module):
    """ConvLSTM module"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        padding,
        stride,
        bias,
        batch_first,
        bidirectional,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.conv_lstm_cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    in_channels, hidden_channels, kernel_size, padding, stride, bias
                )
            ]
        )

        if self.bidirectional:
            self.conv_lstm_cells.append(
                ConvLSTMCell(
                    in_channels, hidden_channels, kernel_size, padding, stride, bias
                )
            )

        self.batch_size = None
        self.seq_len = None
        self.height = None
        self.width = None

    def forward(
        self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # size of x: B, T, C, H, W or T, B, C, H, W
        x = self._check_shape(x)
        hidden_state, cell_state, backward_hidden_state, backward_cell_state = self.init_state(x, state)

        output, hidden_state, cell_state = self._forward(
            self.conv_lstm_cells[0], x, hidden_state, cell_state
        )

        if self.bidirectional:
            x = torch.flip(x, [1])
            backward_output, backward_hidden_state, backward_cell_state = self._forward(
                self.conv_lstm_cells[1], x, backward_hidden_state, backward_cell_state
            )

            output = torch.cat([output, backward_output], dim=-3)
            hidden_state = torch.cat([hidden_state, backward_hidden_state], dim=-1)
            cell_state = torch.cat([cell_state, backward_cell_state], dim=-1)
        return output, (hidden_state, cell_state)

    def _forward(self, lstm_cell, x, hidden_state, cell_state):
        outputs = []
        for time_step in range(self.seq_len):
            x_t = x[:, time_step, :, :, :]
            hidden_state, cell_state = lstm_cell(x_t, hidden_state, cell_state)
            outputs.append(hidden_state.detach())
        output = torch.stack(outputs, dim=1)
        return output, hidden_state, cell_state

    def _check_shape(self, x: Tensor) -> Tensor:
        if self.batch_first:
            batch_size, self.seq_len = x.shape[0], x.shape[1]
        else:
            batch_size, self.seq_len = x.shape[1], x.shape[0]
            x = x.permute(1, 0, 2, 3)
            x = torch.swapaxes(x, 0, 1)

        self.height = x.shape[-2]
        self.width = x.shape[-1]

        dim = len(x.shape)

        if dim == 4:
            x = x.unsqueeze(dim=1)  # increase dimension
            x = x.view(batch_size, self.seq_len, -1, self.height, self.width)
            x = x.contiguous()  # Reassign memory location
        elif dim <= 3:
            raise ValueError(
                f"Got {len(x.shape)} dimensional tensor. Input shape unmatched"
            )

        return x

    def init_state(
        self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]]
    ) -> Tuple[Union[Tensor, Any], Union[Tensor, Any], Optional[Any], Optional[Any]]:
        # If state doesn't enter as input, initialize state to zeros
        backward_hidden_state, backward_cell_state = None, None

        if state is None:
            self.batch_size = x.shape[0]
            hidden_state, cell_state = self._init_state(x.dtype, x.device)

            if self.bidirectional:
                backward_hidden_state, backward_cell_state = self._init_state(
                    x.dtype, x.device
                )
        else:
            if self.bidirectional:
                hidden_state, hidden_state_back = state[0].chunk(2, dim=-1)
                cell_state, cell_state_back = state[1].chunk(2, dim=-1)
            else:
                hidden_state, cell_state = state

        return hidden_state, cell_state, backward_hidden_state, backward_cell_state

    def _init_state(self, dtype, device):
        self.register_buffer(
            "hidden_state",
            torch.zeros(
                (1, self.hidden_channels, self.height, self.width),
                dtype=dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "cell_state",
            torch.zeros(
                (1, self.hidden_channels, self.height, self.width),
                dtype=dtype,
                device=device,
            ),
        )
        return self.hidden_state, self.cell_state
