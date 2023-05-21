import torch
import torch.nn as nn


# Mi to match approximately 3D convolution parameters
class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        Mi = int((in_channels * 3 * 3 * 3 * out_channels) / (3 * 3 * in_channels + 3 * out_channels))
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, Mi, kernel_size=(1, kernel_size[1], kernel_size[2]),
                      stride=(1, stride, stride), padding=(0, 1, 1)),
            nn.BatchNorm3d(Mi),
            nn.ReLU(inplace=True),
            nn.Conv3d(Mi, out_channels, kernel_size=(kernel_size[0], 1, 1),
                      stride=(stride, 1, 1), padding=(1, 0, 0))
        )

    def forward(self, x):
        return self.seq(x)


class Conv2Plus1DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample):
        super().__init__()
        self.downsample_flag = downsample
        if self.downsample_flag:  # down sampling?
            self.downsample = nn.Sequential(
                # perform spatiotemporal down sampling with a 3D convolution through striding
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )
        # Feedforward function
        self.seq = nn.Sequential(
            Conv2Plus1D(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            Conv2Plus1D(out_channels, out_channels, kernel_size, stride=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.seq(x)
        if self.downsample_flag:
            identity = self.downsample(identity)
        # F(x) + x
        x += identity
        return x


class Conv3DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample):
        super().__init__()
        self.downsample_flag = downsample
        if self.downsample_flag:
            self.downsample = nn.Sequential(
                # perform spatiotemporal down sampling with a 3D convolution through striding
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )
        # Feedforward function
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.seq(x)
        if self.downsample_flag:
            identity = self.downsample(identity)
        # F(x) + x
        x += identity
        return x


class Conv2DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample):
        super().__init__()
        self.downsample_flag = downsample
        if self.downsample_flag:
            self.downsample = nn.Sequential(
                # perform spatial down sampling with a 3D convolution through striding
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride)),
                nn.BatchNorm3d(out_channels)
            )
        # Feedforward function
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=(1, stride, stride), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.seq(x)
        if self.downsample_flag:
            identity = self.downsample(identity)
        # F(x) + x
        x += identity
        return x


# Spatiotemporal embedding
class TubeletEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dim, patch_size):
        super().__init__()
        # Create tubelet patches with 3d conv and flatten them
        self.patcher = nn.Conv3d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=patch_size, stride=patch_size, padding='valid')
        self.flatten = nn.Flatten(start_dim=2, end_dim=4)

    def forward(self, x):
        x_patches = self.patcher(x)
        x_flattened = self.flatten(x_patches)
        return x_flattened.permute(0, 2, 1)  # switch to dimension: (batch_size, num_patches, embedding_dimension)


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, num_tokens):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, encoded_tokens):
        batch_size = encoded_tokens.shape[0]

        # Create position indices
        positions = torch.arange(self.num_tokens, dtype=torch.long, device=encoded_tokens.device)
        positions = positions.unsqueeze(0).expand(batch_size, self.num_tokens)

        # Create position embeddings
        position_embedding = nn.Embedding(self.num_tokens, self.embed_dim, device=encoded_tokens.device)
        encoded_positions = position_embedding(positions)
        return encoded_positions


# https://www.learnpytorch.io/08_pytorch_paper_replicating/
class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """

    # Initialize the class with hyperparameters from Table 1
    def __init__(self, embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float = 0.1):  # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)  # does our batch dimension come first?

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,  # query embeddings
                                             key=x,  # key embeddings
                                             value=x,  # value embeddings
                                             need_weights=False)  # do we need the weights or just the layer outputs?
        return attn_output


# https://www.learnpytorch.io/08_pytorch_paper_replicating/
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""

    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self, embedding_dim: int = 768,  # Hidden Size D from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 dropout: float = 0.1):  # Dropout from Table 3 for ViT-Base
        super().__init__()

        # Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,  # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim),  # take back to embedding_dim
            nn.GELU(),
            nn.Dropout(p=dropout)  # "Dropout, when used, is applied after every dense layer.."
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


# https://www.learnpytorch.io/08_pytorch_paper_replicating/
class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""

    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 mlp_dropout: float = 0.1,  # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout: float = 0.1):  # Amount of dropout for attention layers
        super().__init__()

        # Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        # Create MLP block (equation 3)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    def forward(self, x):
        # Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x
        return x


class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),  # spatial
                      stride=(1, stride, stride), padding=(0, padding, padding)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_size[0], 1, 1),  # temporal
                      stride=(stride, 1, 1), padding=(padding, 0, 0))
        )

    def forward(self, x):
        return self.seq(x)


# 3D temporal separable Inception block
class SepInception(nn.Module):
    def __init__(self, in_channels, out_conv1, reduce_conv2, out_conv2,
                 reduce_conv3, out_conv3, out_pool_proj):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_conv1, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_conv1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, reduce_conv2, kernel_size=1, stride=1),
            nn.BatchNorm3d(reduce_conv2),
            nn.ReLU(inplace=True),
            SepConv(reduce_conv2, out_conv2, (3, 3, 3), 1),
            nn.BatchNorm3d(out_conv2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, reduce_conv3, kernel_size=1, stride=1),
            nn.BatchNorm3d(reduce_conv3),
            nn.ReLU(inplace=True),
            SepConv(reduce_conv3, out_conv3, (3, 3, 3), 1),
            nn.BatchNorm3d(out_conv3),
            nn.ReLU(inplace=True)
        )
        self.pool_proj = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_pool_proj, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.pool_proj(x)
        return torch.cat((x1, x2, x3, x4), 1)  # concatenate


# based on: “A Closer Look at Spatiotemporal Convolutions for Action Recognition” by D. Tran et al. (2017).
# and: https://www.tensorflow.org/tutorials/video/video_classification
class R2Plus1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 18 layer ResNet structure
        # Starting convolution based on the paper Appendix
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3)),  # padding to get dimension 54x54 to 56x56
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        # Second 2+1D block
        self.conv2_1 = Conv2Plus1DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = Conv2Plus1DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_2 = nn.ReLU(inplace=True)
        # Third 2+1D block
        self.conv3_1 = Conv2Plus1DResidualBlock(64, 128, (3, 3, 3), 2, True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = Conv2Plus1DResidualBlock(128, 128, (3, 3, 3), 1, False)
        self.relu3_2 = nn.ReLU(inplace=True)
        # Fourth 2+1D block
        self.conv4_1 = Conv2Plus1DResidualBlock(128, 256, (3, 3, 3), 2, True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = Conv2Plus1DResidualBlock(256, 256, (3, 3, 3), 1, False)
        self.relu4_2 = nn.ReLU(inplace=True)
        # Fifth 2+1D block
        self.conv5_1 = Conv2Plus1DResidualBlock(256, 512, (3, 3, 3), 2, True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = Conv2Plus1DResidualBlock(512, 512, (3, 3, 3), 1, False)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        # Final fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Process input through the resnet blocks
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        # Classify with spatio-temporal (global avg) pooling
        x = self.global_avgpool(x)
        x = self.fc(x.flatten(1))
        return x


# based on: “A Closer Look at Spatiotemporal Convolutions for Action Recognition” by D. Tran et al. (2017).
class MC4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 18 layer ResNet structure
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        # Second 3D block
        self.conv2_1 = Conv3DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = Conv3DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_2 = nn.ReLU(inplace=True)
        # Third 3D block
        self.conv3_1 = Conv3DResidualBlock(64, 128, (3, 3, 3), 2, True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = Conv3DResidualBlock(128, 128, (3, 3, 3), 1, False)
        self.relu3_2 = nn.ReLU(inplace=True)
        # First 2D block
        self.conv4_1 = Conv2DResidualBlock(128, 256, (1, 3, 3), 2, True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = Conv2DResidualBlock(256, 256, (1, 3, 3), 1, False)
        self.relu4_2 = nn.ReLU(inplace=True)
        # Second 2D block
        self.conv5_1 = Conv2DResidualBlock(256, 512, (1, 3, 3), 2, True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = Conv2DResidualBlock(512, 512, (1, 3, 3), 1, False)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        # Final fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Process input through the resnet blocks
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        # Classify with spatio-temporal (global avg) pooling
        x = self.global_avgpool(x)
        x = self.fc(x.flatten(1))
        return x


# based on: “ViViT: A Video Vision Transformer” by A. Arnab et al. (2021).
# strong inspiration from: https://keras.io/examples/vision/vivit/ and https://www.learnpytorch.io/08_pytorch_paper_replicating/
class ViViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # Initialize the class with hyperparameters from Table 1 and Table 3 scaled
    def __init__(self, vid_size: int = 112,  # Training resolution from Table 3 in ViViT paper
                 frames: int = 32,
                 in_channels: int = 3,  # Number of channels in input image
                 patch_size=(2, 16, 16),  # Patch size
                 num_transformer_layers: int = 12,  # Layers from Table 1 for ViT-Base
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float = 0.1,  # Dropout for attention projection
                 mlp_dropout: float = 0,  # Dropout for dense/MLP layers
                 embedding_dropout: float = 0,  # Dropout for patch and position embeddings
                 num_classes: int = 3):  # Default for this project
        super().__init__()

        # Calculate number of patches (T/t * H/h * W/w), where the tubelet dimensions are t×h×w
        self.num_patches = int(frames // patch_size[0] * vid_size // patch_size[1] * vid_size // patch_size[2])

        # Create patch embedding layer
        self.patch_embedding = TubeletEmbedding(in_channels=in_channels, embedding_dim=embedding_dim,
                                                patch_size=patch_size)

        # Create learnable position embedding layer
        self.position_embedding = PositionalEncoder(embedding_dim, self.num_patches)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create Transformer Encoder blocks
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout,
                                                                           attn_dropout=attn_dropout) for _ in
                                                   range(num_transformer_layers)])

        # Create classifier head
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        # Create patch embedding
        patch_embeddings = self.patch_embedding(x)
        position_embeddings = self.position_embedding(x)

        # Add position embedding to patch embedding
        x = patch_embeddings + position_embeddings

        # Run embedding dropout
        x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers
        x = self.transformer_encoder(x)

        # Classify with global average pooling
        x = self.layer_norm(x)
        x = self.avgpool(x.permute(0, 2, 1))
        x = x.squeeze(2)
        x = self.fc(x)
        return x


# based on: “Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification.”
#            by S. Xie et al. (2018).
class S3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # S3D network with the Inception V1 as backbone
        self.conv1 = nn.Sequential(
            SepConv(3, 64, (7, 7, 7), 2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            SepConv(64, 192, (3, 3, 3), 1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # First set of inceptions
        self.inc3a = SepInception(192, out_conv1=64, reduce_conv2=96, out_conv2=128, reduce_conv3=16,
                                  out_conv3=32, out_pool_proj=32)
        self.inc3b = SepInception(256, out_conv1=128, reduce_conv2=128, out_conv2=192, reduce_conv3=32,
                                  out_conv3=96, out_pool_proj=64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # Second set of inceptions
        self.inc4a = SepInception(480, out_conv1=192, reduce_conv2=96, out_conv2=208, reduce_conv3=16,
                                  out_conv3=48, out_pool_proj=64)
        self.inc4b = SepInception(512, out_conv1=160, reduce_conv2=112, out_conv2=224, reduce_conv3=24,
                                  out_conv3=64, out_pool_proj=64)
        self.inc4c = SepInception(512, out_conv1=128, reduce_conv2=128, out_conv2=256, reduce_conv3=24,
                                  out_conv3=64, out_pool_proj=64)
        self.inc4d = SepInception(512, out_conv1=112, reduce_conv2=144, out_conv2=288, reduce_conv3=32,
                                  out_conv3=64, out_pool_proj=64)
        self.inc4e = SepInception(528, out_conv1=256, reduce_conv2=160, out_conv2=320, reduce_conv3=32,
                                  out_conv3=128, out_pool_proj=128)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # no padding to match 7x7 dimensions
        # Last set of inceptions
        self.inc5a = SepInception(832, out_conv1=256, reduce_conv2=160, out_conv2=320, reduce_conv3=32,
                                  out_conv3=128, out_pool_proj=128)
        self.inc5b = SepInception(832, out_conv1=384, reduce_conv2=192, out_conv2=384, reduce_conv3=48,
                                  out_conv3=128, out_pool_proj=128)
        # Classification step
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.conv4 = nn.Conv3d(1024, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        # First set of inceptions
        x = self.inc3a(x)
        x = self.inc3b(x)
        x = self.maxpool3(x)
        # Second set of inceptions
        x = self.inc4a(x)
        x = self.inc4b(x)
        x = self.inc4c(x)
        x = self.inc4d(x)
        x = self.inc4e(x)
        x = self.maxpool4(x)
        # Last set of inceptions
        x = self.inc5a(x)
        x = self.inc5b(x)
        x = self.avgpool(x)
        # Classify through convolution and averaging, the predictions are obtained in time
        x = self.conv4(x)
        x = x.squeeze(dim=(3, 4))  # get rid of unnecessary dimension [batch, 3, 3, 1, 1] -> [batch, 3, 3]
        out = x.mean(dim=2)  # [batch, 3, 3] -> [batch, 3]
        return out
