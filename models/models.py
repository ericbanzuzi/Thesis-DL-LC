import torch.nn as nn
import torch
from torchinfo import summary


class Conv2Plus1DFirst(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64)
        )


class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2Plus1D, self).__init__()
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
        super(Conv2Plus1DResidualBlock, self).__init__()
        self.downsample_flag = downsample
        if downsample:
            self.downsample = nn.Sequential(
                # perform down sampling with a 3D convolution
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )

        self.seq = nn.Sequential(
            Conv2Plus1D(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            Conv2Plus1D(out_channels, out_channels, kernel_size, stride=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.seq(x)
        if self.downsample_flag:
            residual = self.downsample(residual)
        out += residual
        return out


class Conv3DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample):
        super(Conv3DResidualBlock, self).__init__()
        self.downsample_flag = downsample
        if downsample:
            self.downsample = nn.Sequential(
                # perform down sampling with a 3D convolution
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )

        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.seq(x)
        if self.downsample_flag:
            residual = self.downsample(residual)
        out += residual
        return out


class Conv2DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample):
        super(Conv2DResidualBlock, self).__init__()
        self.downsample_flag = downsample
        if downsample:
            self.downsample = nn.Sequential(
                # perform down sampling with a 3D convolution
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride)),
                nn.BatchNorm3d(out_channels)
            )

        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=(1, stride, stride), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.seq(x)
        if self.downsample_flag:
            residual = self.downsample(residual)
        out += residual
        return out


class TubeletEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dim, patch_size):
        super(TubeletEmbedding, self).__init__()
        self.patcher = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding='valid'
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=4)

    def forward(self, x):
        x_patches = self.patcher(x)
        x_flattened = self.flatten(x_patches)
        return x_flattened.permute(0, 2, 1)


# 1. Create a class that inherits from nn.Module
class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """

    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float = 0):  # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)  # does our batch dimension come first?

    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,  # query embeddings
                                             key=x,  # key embeddings
                                             value=x,  # value embeddings
                                             need_weights=False)  # do we need the weights or just the layer outputs?
        return attn_output


class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""

    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim: int = 768,  # Hidden Size D from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 dropout: float = 0.1):  # Dropout from Table 3 for ViT-Base
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,  # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim),  # take back to embedding_dim
            nn.Dropout(p=dropout)  # "Dropout, when used, is applied after every dense layer.."
        )

    # Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""

    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 mlp_dropout: float = 0.1,  # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout: float = 0):  # Amount of dropout for attention layers
        super().__init__()

        # Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        # Create MLP block (equation 3)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    # Create a forward() method
    def forward(self, x):
        # Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x


# based on: “A Closer Look at Spatiotemporal Convolutions for Action Recognition” by D. Tran et al. (2017).
# and: https://www.tensorflow.org/tutorials/video/video_classification
class R2Plus1D(nn.Module):
    def __init__(self, num_classes):
        super(R2Plus1D, self).__init__()
        self.conv1 = Conv2Plus1DFirst()
        self.relu1 = nn.ReLU(inplace=True)
        # First 2+1D block
        self.conv2_1 = Conv2Plus1DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = Conv2Plus1DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_2 = nn.ReLU(inplace=True)
        # Second 2+1D block
        self.conv3_1 = Conv2Plus1DResidualBlock(64, 128, (3, 3, 3), 2, True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = Conv2Plus1DResidualBlock(128, 128, (3, 3, 3), 1, False)
        self.relu3_2 = nn.ReLU(inplace=True)
        # Third 2+1D block
        self.conv4_1 = Conv2Plus1DResidualBlock(128, 256, (3, 3, 3), 2, True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = Conv2Plus1DResidualBlock(256, 256, (3, 3, 3), 1, False)
        self.relu4_2 = nn.ReLU(inplace=True)
        # Fourth 2+1D block
        self.conv5_1 = Conv2Plus1DResidualBlock(256, 512, (3, 3, 3), 2, True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = Conv2Plus1DResidualBlock(512, 512, (3, 3, 3), 1, False)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Final fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
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
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# based on: “A Closer Look at Spatiotemporal Convolutions for Action Recognition” by D. Tran et al. (2017).
class MC4(nn.Module):
    def __init__(self, num_classes):
        super(MC4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64)
        )
        self.relu1 = nn.ReLU(inplace=True)
        # First 3D block
        self.conv2_1 = Conv3DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = Conv3DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_2 = nn.ReLU(inplace=True)
        # Second 3D block
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

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Final fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
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
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ViViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""

    # Initialize the class with hyperparameters from Table 1 and Table 3 for 224×224 input
    def __init__(self,
                 vid_size: int = 224,  # Training resolution from Table 3 in ViViT paper
                 frames: int = 32,
                 in_channels: int = 3,  # Number of channels in input image
                 patch_size=(2, 16, 16),  # Patch size
                 num_transformer_layers: int = 12,  # Layers from Table 1 for ViT-Base
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float = 0,  # Dropout for attention projection
                 mlp_dropout: float = 0.1,  # Dropout for dense/MLP layers
                 embedding_dropout: float = 0.1,  # Dropout for patch and position embeddings
                 num_classes: int = 3):  # Default for ImageNet but can customize this
        super().__init__()  # don't forget the super().__init__()!

        # Calculate number of patches (T/t * H/h * W/w), where the tubelet dimensions are t×h×w
        self.num_patches = int(frames/patch_size[0] * vid_size/patch_size[1] * vid_size/patch_size[2])

        # Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim),
                                               requires_grad=True)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create patch embedding layer
        self.patch_embedding = TubeletEmbedding(in_channels=in_channels, embedding_dim=embedding_dim,
                                                patch_size=patch_size)

        # Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
        #                                                                    num_heads=num_heads,
        #                                                                    mlp_size=mlp_size,
        #                                                                    mlp_dropout=mlp_dropout) for _ in
        #                                            range(num_transformer_layers)])
        self.transformer_encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=embedding_dim,  # Hidden size D from Table 1 for ViT-Base
                                                                              nhead=num_heads,  # Heads from Table 1 for ViT-Base
                                                                              dim_feedforward=mlp_size,  # MLP size from Table 1 for ViT-Base
                                                                              dropout=mlp_dropout,  # Amount of dropout for dense layers from Table 3 for ViT-Base
                                                                              activation="gelu",  # GELU non-linear activation
                                                                              batch_first=True,  # Do our batches come first?
                                                                              norm_first=True)
                                                   for _ in range(num_transformer_layers)])  # Normalize first or after MSA/MLP layers?

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]

        # Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1,
                                                  -1)  # "-1" means to infer the dimension (try this line on its own)
        # Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # Add position embedding to patch embedding (equation 1)
        x = self.position_embedding + x

        # Run embedding dropout
        x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0])  # run on each sample in a batch at 0 index

        return x


# for visualizing architectures
if __name__ == '__main__':
    model = MC4(num_classes=3)
    PATCH_SIZE = (2, 16, 16)
    h = 28
    NUM_PATCHES = int((224 * 224) / 16 ** 2)
    print(NUM_PATCHES)
    # (112 // PATCH_SIZE[0]) ** 2
    dim = PATCH_SIZE[1] * PATCH_SIZE[2] * 3
    print(dim)
    model = TubeletEmbedding(embedding_dim=dim, patch_size=PATCH_SIZE)
    model = ViViT()
    # print(summary(model, input_size=(3, 32, 224, 224)))
    # Print a summary of our custom ViT model using torchinfo (uncomment for actual output)
    summary(model=model,
            input_size=(32, 3, 32, 224, 224),  # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )