from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torchvision.models import resnet34
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def get_model(arch: str, n_classes: int, input_shape: Tuple[int]):
    logger.info(f"Get model: {arch=}, {n_classes=}, {input_shape=}")
    if arch == "simple-cnn-1d":
        return SimpleCNN1D(n_classes=n_classes, input_shape=input_shape)
    elif arch == "simple-cnn-2d":
        return SimpleCNN2D(n_classes=n_classes, input_shape=input_shape)
    elif arch == "resnet34":
        return ResNet34(n_classes=n_classes, input_shape=input_shape)
    elif arch == "eegnet":
        return EEGNet(n_classes=n_classes, input_shape=input_shape)
    elif arch == "convmixer":
        return ConvMixer(input_shape=input_shape, dim=768, depth=32, kernel_size=7, patch_size=7, n_classes=n_classes)
    elif arch == "combined-cnn":
        return  CombinedCNN(n_classes=n_classes, input_shape=input_shape)
    elif arch == "combined-cnn-no-attention":
        return  CombinedCNN_NoAttention(n_classes=n_classes, input_shape=input_shape)
    elif arch == "combined-cnn-no-attention-tiny":
        return  CombinedCNN_NoAttention_tiny(n_classes=n_classes, input_shape=input_shape)
    elif arch == "combined-cnn-tiny":
        return  CombinedCNN_tiny(n_classes=n_classes, input_shape=input_shape)
    #elif arch == "transformer":
     #   return  HybridCNNTransformer(n_classes=n_classes, input_shape=input_shape)
    else:
        raise ValueError()


# See https://github.com/locuslab/convmixer/blob/main/convmixer.py
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, input_shape, dim, depth, kernel_size=9, patch_size=7, n_classes=2):
        super().__init__()
        input_chans = input_shape[0]
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_chans, affine=False),
            nn.Conv1d(input_chans, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"), nn.GELU(), nn.BatchNorm1d(dim)
                        )
                    ),
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm1d(dim),
                )
                for _ in range(depth)
            ],
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, n_classes),
            nn.LogSoftmax(-1),
        )

    def forward(self, data):
        return self.net(data)


class ResNet34(nn.Module):
    def __init__(self, n_classes: int, input_shape: Tuple[int], pretrained=True):
        super().__init__()
        self.net = resnet34(pretrained=pretrained, progress=False)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Calculate the size of the output from the penultimate layer
        with torch.no_grad():
            self.net.fc = nn.Identity()  # Temporarily remove the final layer
            # Create a dummy variable of the right input shape to pass through the network
            dummy_data = torch.zeros(1, 1, *input_shape)
            features_dim = self.net(dummy_data).shape[1]

        # Now, recreate the final fully connected layer with the correct number of input features
        self.net.fc = nn.Linear(features_dim, n_classes)

    def forward(self, data):
        data = data.unsqueeze(1)  # Add a channel dimension
        return F.log_softmax(self.net(data), dim=-1)


def _CNN_N_DIM(n_dim, input_channels):
    if n_dim == 1:
        conv, bn = nn.Conv1d, nn.BatchNorm1d
    elif n_dim == 2:
        conv, bn = nn.Conv2d, nn.BatchNorm2d
    else:
        raise ValueError("Only 1D and 2D CNNs are supported")

    return nn.Sequential(
        conv(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
        bn(32),
        nn.SiLU(inplace=True),
        conv(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
        bn(64),
        nn.SiLU(inplace=True),
        conv(64, 128, kernel_size=3, stride=2, bias=False),
        bn(128),
        nn.SiLU(inplace=True),
        conv(128, 256, kernel_size=3, stride=2, bias=False),
        bn(256),
        nn.SiLU(inplace=True),
        conv(256, 512, kernel_size=3, stride=2, bias=False),
        nn.Flatten(),
    )


class _SimpleCNNXD(nn.Module):
    def __init__(self, n_dim: int, n_classes: int, input_shape: Tuple[int]):
        super().__init__()
        self.net = _CNN_N_DIM(n_dim=n_dim, input_channels=input_shape[0])
        _hidden_size = self.net(torch.zeros(1, *input_shape)).numel()
        self.classifier = nn.Sequential(
            nn.Linear(_hidden_size, n_classes),
            nn.LogSoftmax(-1),
        )

    def forward(self, data):
        return self.classifier(self.net(data))


class SimpleCNN2D(_SimpleCNNXD):
    """Expects data with shape (batch, channels, H, W).
    Suitable for EEG data in time-frequency domain."""

    def __init__(self, n_classes: int, input_shape: Tuple[int]):
        input_shape = (1, *input_shape)
        super().__init__(n_dim=2, n_classes=n_classes, input_shape=input_shape)

    def forward(self, data):
        return super().forward(data.unsqueeze(1))


class SimpleCNN1D(_SimpleCNNXD):
    """Expects data with shape (batch, channels, time).
    Suitable for EEG data in time domain."""

    def __init__(self, n_classes: int, input_shape: Tuple[int]):
        super().__init__(n_dim=1, n_classes=n_classes, input_shape=input_shape)


class EEGNet(nn.Module):
    """
    Attempt to reproduce model from: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py#L55
    NOTE - Some differences from original EEGNet:
    - kernel_constraints. https://discuss.pytorch.org/t/kernel-constraint-similar-to-the-one-implemented-in-keras/49936
        (could use pytorch parametrizations, geotorch, or similar)
    - Separable Conv may not be exactly equivalent. Need to choose when to increase channels:
        F1 -> F1, F1 -> F2   VS   F1 -> F2, F2 -> F2
    - Fixed latent dimension size using a final conv
    """

    def __init__(
        self,
        n_classes: int,
        input_shape: Tuple[int],
        feature_dim: int = 64,
        dropout_rate=0.5,
        F1=8,
        F2=16,
    ):
        super().__init__()
        input_channels, input_time_length = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 16), padding=8, bias=False),  # Originally 64, padding=32
            nn.BatchNorm2d(F1),
            # NOTE - groups == in_chan corresponds to "Depthwise Conv"
            nn.Conv2d(F1, F1, (input_channels, 1), bias=False, groups=F1),
            nn.BatchNorm2d(F1),
            nn.SiLU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
            # NOTE - two separate convolutions to implement one "Separable Conv"
            nn.Conv2d(F1, F2, (1, 8), bias=False, groups=1),  # Originally both 16 instead of 8
            nn.Conv2d(F2, F2, (8, 1), bias=False, groups=F2),
            nn.BatchNorm2d(F2),
            nn.SiLU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        shape_after_conv = self.conv(torch.zeros(1, 1, input_channels, input_time_length)).shape
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(F2, feature_dim, (shape_after_conv[2], shape_after_conv[3]), bias=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, n_classes),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add dummy channel dimension
        return self.classifier(self.feature_reduction(self.conv(x)))
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, scale=None):
        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)
        return output, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, d_model=64, dropout=0.1):
        super().__init__()
        self.dim_per_head = d_model // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(d_model, self.num_heads * self.dim_per_head)
        self.linear_v = nn.Linear(d_model, self.num_heads * self.dim_per_head)
        self.linear_q = nn.Linear(d_model, self.num_heads * self.dim_per_head)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = query.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size, -1, num_heads, dim_per_head)
        value = value.view(batch_size, -1, num_heads, dim_per_head)
        query = query.view(batch_size, -1, num_heads, dim_per_head)

        scaled_attention, attention = self.dot_product_attention(query, key, value, scale=np.power(dim_per_head, 0.5))
        scaled_attention = scaled_attention.view(batch_size, -1, num_heads * dim_per_head)

        output = self.linear_final(scaled_attention)
        output = self.dropout(output)
        output = self.layer_norm(output + query.view(batch_size, -1, num_heads * dim_per_head))
        return output, attention

class CombinedCNN_NoAttention(nn.Module):
    def __init__(self, n_classes: int, input_shape: Tuple[int]):
        super(CombinedCNN_NoAttention, self).__init__() 

        # 2D Convolutional layers for spatial feature extraction
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, (input_shape[0], 3), padding=(0, 1)),  # channels x time -> 16 features
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d((1, 2)),  # Reduce time dimension
            
            # Additional 2D convolutional layers
            nn.Conv2d(16, 32, (1, 3), padding=(0, 1)),  # Increase features to 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d((1, 2)),
            
            # You can continue adding more layers here...
        )

        # Calculate the output size after the spatial layers
        spatial_output_size = self._get_spatial_output(input_shape)

        # 1D Convolutional layers for temporal feature extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # Apply 1D convolutions on the temporal dimension
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # Increase features to 128
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # Additional 1D convolutional layers
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # Increase features to 256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling

            # You can continue adding more layers here...
        )

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, n_classes),  # Adjust the input features according to the last layer's output
            nn.LogSoftmax(dim=-1)
        )

    def _get_spatial_output(self, input_shape):
        # Helper function to calculate the output size of the spatial layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            output = self.spatial(dummy_input)
        return output.size()[2:]

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.spatial(x)  # Apply 2D convolutions
        x = x.squeeze(2)  # Remove the one-dimensional spatial axis
        x = self.temporal(x)  # Apply 1D convolutions
        x = self.fc(x)  # Classify
        return x




class CombinedCNN_NoAttention_tiny(nn.Module):
    def __init__(self, n_classes: int, input_shape: Tuple[int]):
        super(CombinedCNN_NoAttention_tiny, self).__init__()  # Correct class name here

        # 2D Convolutional layers for spatial feature extraction
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, (input_shape[0], 3), padding=(0, 1)),  # channels x time -> 16 features
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d((1, 2))  # Reduce time dimension
        )

        # Calculate the output size after the spatial layers
        spatial_output_size = self._get_spatial_output(input_shape)

        # 1D Convolutional layers for temporal feature extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # First 1D convolution
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # Additional 1D convolution
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, n_classes),  # Adjust the input features according to the last layer's output
            nn.LogSoftmax(dim=-1)
        )

    def _get_spatial_output(self, input_shape):
        # Helper function to calculate the output size of the spatial layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            output = self.spatial(dummy_input)
        return output.size()[2:]

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.spatial(x)  # Apply 2D convolutions
        x = x.squeeze(2)  # Remove the one-dimensional spatial axis
        x = self.temporal(x)  # Apply 1D convolutions
        x = self.fc(x)  # Classify
        return x


class CombinedCNN_tiny(nn.Module):
    def __init__(self, n_classes: int, input_shape: Tuple[int]):
        super(CombinedCNN_tiny, self).__init__()

        # 2D Convolutional layers for spatial feature extraction
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, (input_shape[0], 3), padding=(0, 1)),  # channels x time -> 16 features
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d((1, 2))  # Reduce time dimension
        )

        # Calculate the output size after the spatial layers
        spatial_output_size = self._get_spatial_output(input_shape)

        # 1D Convolutional layers for temporal feature extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # First 1D convolution
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # Additional 1D convolution
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

        self.d_model = 64  # Define the model dimension for attention layer
        self.num_heads = 8  # Define the number of heads for MultiHeadAttention
        self.attention = MultiHeadAttention(num_heads=self.num_heads, d_model=self.d_model)

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.d_model, n_classes),  # Adjusted for attention layer output
            nn.LogSoftmax(dim=-1)
        )

    def _get_spatial_output(self, input_shape):
        # Helper function to calculate the output size of the spatial layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            output = self.spatial(dummy_input)
        return output.size()[2:]

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.spatial(x)  # Apply 2D convolutions
        x = x.squeeze(2)  # Remove the one-dimensional spatial axis
        x = self.temporal(x)  # Apply 1D convolutions

        # Apply attention
        x = x.transpose(1, 2)  # Reshape for attention layer: [batch, seq_len, features]
        x, _ = self.attention(x, x, x)  # Self-attention

        x = self.fc(x)  # Classify
        return x
    
class CombinedCNN(nn.Module):
    def __init__(self, n_classes: int, input_shape: Tuple[int], d_model=64, num_heads=8, dropout=0.1):
        super(CombinedCNN, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # 2D Convolutional layers for spatial feature extraction
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, (input_shape[0], 3), padding=(0, 1)),  # channels x time -> 16 features
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d((1, 2)),  # Reduce time dimension
            
            # Additional 2D convolutional layers
            nn.Conv2d(16, 32, (1, 3), padding=(0, 1)),  # Increase features to 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d((1, 2)),
            
            # You can continue adding more layers here...
        )

        # Calculate the output size after the spatial layers
        spatial_output_size = self._get_spatial_output(input_shape)

        # 1D Convolutional layers for temporal feature extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # Apply 1D convolutions on the temporal dimension
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # Increase features to 128
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # Additional 1D convolutional layers
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # Increase features to 256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling

            # You can continue adding more layers here...
        )
        # MultiHeadAttention layer
        self.attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)

        # A new linear layer to ensure the output of the attention layer is the correct size for the classification layer
        self.match_d_model = nn.Linear(256, d_model)  # Assuming 256 is the number of features from the last 1D conv layer

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, n_classes),  # Adjust the input features according to the last layer's output
            nn.LogSoftmax(dim=-1)
        )

    def _get_spatial_output(self, input_shape):
        # Helper function to calculate the output size of the spatial layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            output = self.spatial(dummy_input)
        return output.size()[2:]

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.spatial(x)  # Apply 2D convolutions
        x = x.squeeze(2)  # Remove the one-dimensional spatial axis
        x = self.temporal(x)  # Apply 1D convolutions
        # Reshape & Transpose for MultiHeadAttention
        x = x.transpose(1, 2)  # B x C x L -> B x L x C
        # Transform the input to match the d_model of the attention mechanism
        x = self.match_d_model(x)
        # Apply MultiHeadAttention
        x, _ = self.attention(x, x, x)  # Apply Self-Attention
        # Apply the fully connected layer for classification
        x = self.fc(x)  
        return x



class HybridCNNTransformer(nn.Module):
    def __init__(self, n_classes: int, input_shape: Tuple[int], nhead=2, d_model=16, num_layers=1, dropout=0.5):
        super(HybridCNNTransformer, self).__init__()

        self.d_model = d_model  # Ensure this is the same as the number of features after the CNN

        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(1, 16, (input_shape[0], 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 32, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d((1, 2)),

            nn.Flatten(),
            
            # Ensure the output of CNN is the correct size for the transformer
            nn.Linear(input_shape[1]//4 * 32, d_model)  # Adjust the input features to match d_model
        )

        self.positional_encoding = PositionalEncoding(d_model)
        
        transformer_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(transformer_layers, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # Convert to (Batch, Channel, Time, Features)
        x = x.unsqueeze(1)
        
        # Apply CNN for spatial feature extraction
        spatial_features = self.spatial_cnn(x)
        
        # Prepare data for the transformer
        # No need for view as the last layer of CNN should output the correct shape
        spatial_features = self.positional_encoding(spatial_features)
        
        # Apply Transformer for temporal feature extraction
        transformer_output = self.transformer_encoder(spatial_features)
        
        # Take the last time step for classification
        out = transformer_output[:, -1, :]
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class CombinedCNN(nn.Module):
    def __init__(self, n_classes: int, input_shape: Tuple[int]):
        super(CombinedCNN, self).__init__()

        # Assuming input_shape is (channels, length)
        # Convert the input to have the correct shape for 1D convolutions
        self.temporal = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_shape[0], 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 2
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 4
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1) # Global average pooling
            )
                # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, n_classes),  # Now n_classes is defined in the scope
            nn.LogSoftmax(dim=-1)
        )
    def forward(self, x):
        # Reshape input data to (batch, channels, length)
        if x.dim() == 4:  # If input is 4D (batch, channels, height, width), flatten spatial dimensions
            x = x.view(x.size(0), x.size(1), -1)
        elif x.dim() == 3:  # If input is already 3D, use as is
            pass
        else:
            raise ValueError("Expected input to be 3D or 4D, got shape: {}".format(x.shape))

        x = self.temporal(x)  # Apply 1D convolutions
        x = self.fc(x)  # Classify
        return x