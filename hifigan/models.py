import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

# Leaky ReLU slope constant for activation function
LRELU_SLOPE = 0.1

# Function to initialize weights with a normal distribution
def init_weights(m, mean=0.0, std=0.01):
    """
    Initialize weights for convolutional layers with a normal distribution.
    Applies to layers with "Conv" in their classname.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

# Helper function to calculate padding for convolution with dilation
def get_padding(kernel_size, dilation=1):
    """
    Calculate padding to ensure output dimensions remain consistent.
    """
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        """
        Residual block for processing input with convolution layers and skip connections.
        This block has multiple dilated convolutions to increase receptive field.
        """
        super(ResBlock, self).__init__()

        # First convolutional layers with specified dilations
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        # Initialize weights for conv layers in convs1
        self.convs1.apply(init_weights)

        # Second set of convolutional layers with standard dilation of 1
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        
        # Initialize weights for conv layers in convs2
        self.convs2.apply(init_weights)

    def forward(self, x):
        """
        Forward pass through each convolution layer in convs1 and convs2 with residual addition.
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x    # Residual addition
        return x

    def remove_weight_norm(self):
        """
        Removes weight normalization for each convolution layer.
        Can be useful after training to speed up inference.
        """
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        """
        Generator model composed of upsampling layers followed by residual blocks.
        Used in vocoder models to generate audio from mel-spectrograms.
        """
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        
        # Initial convolution layer to process input with 80 channels
        self.conv_pre = weight_norm(
            Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock

        # Upsampling layers (ConvTranspose1d)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2 ** i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Residual blocks after each upsampling layer
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d))

        # Final post-processing convolution
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        
        # Apply weight initialization to upsampling and post-processing layers
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
        Forward pass through the Generator model.
        Processes input with upsampling, residual blocks, and final post-convolution.
        """
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            
            # Apply residual blocks and combine their outputs
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels    # Average outputs of residual blocks
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)    # Ensure output is in range [-1, 1]

        return x

    def remove_weight_norm(self):
         """
        Removes weight normalization from all layers in the model.
        Useful for optimizing model inference speed after training.
        """
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

# Possible Adjustments:
# 1. **Change LRELU_SLOPE**: Adjusting `LRELU_SLOPE` can change the activation dynamics.
#    Higher values can lead to more aggressive activation, while lower values create a more subtle effect.
#
# 2. **Experiment with different kernel and dilation sizes**: `kernel_size` and `dilation` in `ResBlock` 
#    affect the receptive field and model's capacity to capture patterns in the input. 
#    Larger values capture broader patterns but can reduce local detail.
#
# 3. **Varying upsample parameters**: Changing `h.upsample_rates` and `h.upsample_kernel_sizes` in `Generator` 
#    affects the scale and smoothness of upsampling, impacting the fidelity of generated audio.
#
# 4. **Remove final tanh activation**: Removing `torch.tanh(x)` in the `Generator` may be beneficial 
#    in certain tasks if outputs outside [-1, 1] are acceptable. However, this requires careful handling 
#    to avoid clipping or distortion.
#
# 5. **Weight normalization removal**: The `remove_weight_norm` function removes weight normalization, 
#    which is beneficial for faster inference. However, it may slightly affect model outputs, so this 
#    should be done cautiously and validated.
