# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import math
import os
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from datetime import datetime
# from version import __version__  # Commented out for local testing

# Third-party imports
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from accelerate import Accelerator
from ema_pytorch import EMA
import wandb
from tqdm.auto import tqdm
import pdb

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Named tuple for model predictions containing noise and start predictions
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def exists(x):
    """
    Check if a value exists (is not None).
    
    Args:
        x: Value to check
        
    Returns:
        bool: True if x is not None, False otherwise
    """
    return x is not None


def default(val, d):
    """
    Return the first value if it exists, otherwise return the default value.
    
    Args:
        val: Primary value to check
        d: Default value or callable that returns the default value
        
    Returns:
        The primary value if it exists, otherwise the default value
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    """
    Identity function that returns the input unchanged.
    
    Args:
        t: Input tensor
        *args: Additional positional arguments (ignored)
        **kwargs: Additional keyword arguments (ignored)
        
    Returns:
        The input tensor unchanged
    """
    return t


def cycle(dl):
    """
    Create an infinite iterator that cycles through a dataloader.
    
    Args:
        dl: DataLoader to cycle through
        
    Yields:
        Data batches from the dataloader in a continuous loop
    """
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    """
    Check if a number has an integer square root.
    
    Args:
        num: Number to check
        
    Returns:
        bool: True if the number has an integer square root, False otherwise
    """
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    """
    Divide a number into groups of a specified divisor.
    
    Args:
        num: Total number to divide
        divisor: Size of each group
        
    Returns:
        list: List of group sizes, with the last group containing the remainder
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    """
    Convert an image to a specified type if it's not already that type.
    
    Args:
        img_type: Target image type
        image: PIL Image to convert
        
    Returns:
        PIL Image: Converted image or original if already correct type
    """
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_to_neg_one_to_one(img):
    """
    Normalize image values from [0, 1] to [-1, 1].
    
    Args:
        img: Input tensor with values in [0, 1]
        
    Returns:
        Tensor: Normalized tensor with values in [-1, 1]
    """
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """
    Unnormalize tensor values from [-1, 1] to [0, 1].
    
    Args:
        t: Input tensor with values in [-1, 1]
        
    Returns:
        Tensor: Unnormalized tensor with values in [0, 1]
    """
    return (t + 1) * 0.5


def uniform(shape, device):
    """
    Create a tensor of uniform random values between 0 and 1.
    
    Args:
        shape: Shape of the tensor to create
        device: Device to place the tensor on
        
    Returns:
        Tensor: Uniform random tensor with values in [0, 1]
    """
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    """
    Create a boolean mask tensor with specified probability of True values.
    
    Args:
        shape: Shape of the mask tensor
        prob: Probability of True values (between 0 and 1)
        device: Device to place the tensor on
        
    Returns:
        Tensor: Boolean mask tensor
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

# =============================================================================
# DATASET CLASS
# =============================================================================

class Dataset1D(Dataset):
    """
    Simple 1D dataset wrapper for tensor data.
    
    This class provides a standard PyTorch Dataset interface for 1D tensor data,
    allowing easy integration with DataLoader for batch processing.
    """
    
    def __init__(self, tensor: Tensor):
        """
        Initialize the dataset with a tensor.
        
        Args:
            tensor: Input tensor containing the data
        """
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.tensor)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tensor: Sample at the specified index
        """
        return self.tensor[idx].clone()

# =============================================================================
# SMALL HELPER MODULES
# =============================================================================

class Residual(nn.Module):
    """
    Residual connection wrapper.
    
    This module adds a residual connection to any function f, such that
    the output is f(x) + x. This is commonly used in neural networks
    to help with gradient flow and training stability.
    """
    
    def __init__(self, fn):
        """
        Initialize the residual wrapper.
        
        Args:
            fn: Function to wrap with residual connection
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor
            *args: Additional positional arguments for fn
            **kwargs: Additional keyword arguments for fn
            
        Returns:
            Tensor: f(x) + x
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None, scale_factor=2):
    """
    Create an upsampling module using nearest neighbor interpolation and convolution.
    
    Args:
        dim: Input dimension
        dim_out: Output dimension (defaults to dim if None)
        scale_factor: Upsampling factor (default: 2)
        
    Returns:
        nn.Sequential: Upsampling module
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )


def Final_upsample_to_target_length(dim_out, dim_in, target_length):
    """
    Create a final upsampling module to reach a specific target length.
    
    Args:
        dim_out: Output dimension
        dim_in: Input dimension
        target_length: Target sequence length
        
    Returns:
        nn.Sequential: Upsampling module to target length
    """
    return nn.Sequential(
        nn.Upsample(size=target_length, mode='nearest'),
        nn.Conv1d(dim_out, dim_in, 3, padding=1)
    )


def Downsample(dim, dim_out=None, kernel=4):
    """
    Create a downsampling module using strided convolution.
    
    Args:
        dim: Input dimension
        dim_out: Output dimension (defaults to dim if None)
        kernel: Convolution kernel size (default: 4)
        
    Returns:
        nn.Conv1d: Downsampling convolution layer
    """
    return nn.Conv1d(dim, default(dim_out, dim), kernel, 2, 1)


class RMSNorm(nn.Module):
    """
    Root Mean Square normalization layer.
    
    This normalization technique normalizes the input using the RMS of the
    feature dimension, which can be more stable than standard normalization.
    """
    
    def __init__(self, dim):
        """
        Initialize RMS normalization.
        
        Args:
            dim: Feature dimension to normalize over
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (batch, features, length)
            
        Returns:
            Tensor: Normalized tensor
        """
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper.
    
    This module applies normalization before passing the input to a function,
    which can improve training stability and convergence.
    """
    
    def __init__(self, dim, fn):
        """
        Initialize pre-normalization wrapper.
        
        Args:
            dim: Feature dimension for normalization
            fn: Function to apply after normalization
        """
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        """
        Apply normalization followed by the wrapped function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Output of fn(normalize(x))
        """
        x = self.norm(x)
        return self.fn(x)

# =============================================================================
# SINUSOIDAL POSITIONAL EMBEDDINGS
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings for sequence modeling.
    
    This module generates sinusoidal positional embeddings that can be added
    to input sequences to provide position information to the model.
    """
    
    def __init__(self, dim, theta=10000):
        """
        Initialize sinusoidal positional embeddings.
        
        Args:
            dim: Dimension of the positional embeddings
            theta: Frequency parameter for the sinusoids (default: 10000)
        """
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """
        Generate sinusoidal positional embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, 1) containing positions
            
        Returns:
            Tensor: Positional embeddings of shape (batch_size, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Random or learned sinusoidal positional embeddings.
    
    This module provides an alternative to fixed sinusoidal embeddings by
    allowing the positional encoding to be either random or learned during
    training. Based on the implementation by @crowsonkb.
    """
    
    def __init__(self, dim, is_random=False):
        """
        Initialize random or learned sinusoidal positional embeddings.
        
        Args:
            dim: Dimension of the positional embeddings (must be even)
            is_random: If True, use random weights; if False, learn weights
        """
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        """
        Generate random or learned sinusoidal positional embeddings.
        
        Args:
            x: Input tensor of shape (batch_size,) containing positions
            
        Returns:
            Tensor: Positional embeddings of shape (batch_size, dim + 1)
        """
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# =============================================================================
# BUILDING BLOCK MODULES
# =============================================================================

class Block(nn.Module):
    """
    Basic building block with convolution, normalization, and activation.
    
    This module implements a standard neural network block consisting of
    a convolution layer, group normalization, and SiLU activation.
    """
    
    def __init__(self, dim, dim_out, groups=8):
        """
        Initialize the block.
        
        Args:
            dim: Input dimension
            dim_out: Output dimension
            groups: Number of groups for group normalization (default: 8)
        """
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """
        Forward pass through the block.
        
        Args:
            x: Input tensor
            scale_shift: Optional scale and shift parameters for conditional normalization
            
        Returns:
            Tensor: Output after convolution, normalization, and activation
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """
    Residual block with time and class conditioning.
    
    This module implements a residual block that can be conditioned on both
    time embeddings and class embeddings, commonly used in diffusion models.
    """
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        """
        Initialize the residual block.
        
        Args:
            dim: Input dimension
            dim_out: Output dimension
            time_emb_dim: Dimension of time embeddings (optional)
            classes_emb_dim: Dimension of class embeddings (optional)
            groups: Number of groups for group normalization (default: 8)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            time_emb: Time embeddings (optional)
            class_emb: Class embeddings (optional)
            
        Returns:
            Tensor: Output after residual block processing
        """
        scale_shift = None

        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            try:
                cond_emb = self.mlp(cond_emb)
            except:
                print("wrong size for conditional embedding!")
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')

            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    """
    Linear attention mechanism for efficient sequence processing.
    
    This module implements a linear attention mechanism that reduces the
    computational complexity from O(n²) to O(n) for sequence length n.
    """
    
    def __init__(self, dim, heads=4, dim_head=32):
        """
        Initialize linear attention.
        
        Args:
            dim: Input dimension
            heads: Number of attention heads (default: 4)
            dim_head: Dimension per attention head (default: 32)
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        Apply linear attention.
        
        Args:
            x: Input tensor of shape (batch, dim, seq_len)
            
        Returns:
            Tensor: Output after linear attention
        """
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h e n -> b (h e) n')
        return self.to_out(out)

class Attention(nn.Module):
    """
    Standard multi-head attention mechanism.
    
    This module implements the standard scaled dot-product attention
    mechanism used in transformer architectures.
    """
    
    def __init__(self, dim, heads=4, dim_head=32):
        """
        Initialize multi-head attention.
        
        Args:
            dim: Input dimension
            heads: Number of attention heads (default: 4)
            dim_head: Dimension per attention head (default: 32)
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, dim, seq_len)
            
        Returns:
            Tensor: Output after attention
        """
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h d i', attn, v)

        out = rearrange(out, 'b h d n -> b (h d) n')
        return self.to_out(out)

# =============================================================================
# MODEL
# =============================================================================

class Unet1D(nn.Module):
    """
    1D U-Net architecture for diffusion models with classifier-free guidance.
    
    This module implements a 1D U-Net architecture that can be conditioned on
    class embeddings and supports classifier-free guidance for improved generation
    quality. The architecture follows the standard U-Net pattern with skip
    connections and attention mechanisms.
    """
    
    def __init__(
            self,
            dim,
            class_dim,
            seq_length,
            cond_drop_prob=0.5,
            mask_val=0.0,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            embed_class_layers_dims=(64, 64),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            sinusoidal_pos_emb_theta=10000,
            attn_dim_head=32,
            attn_heads=4,
    ):
        """
        Initialize the 1D U-Net architecture.
        
        Args:
            dim: Base dimension for the network
            class_dim: Dimension of class embeddings
            seq_length: Length of input sequences
            cond_drop_prob: Probability of dropping conditioning during training
            mask_val: Value to use for masked conditioning
            init_dim: Initial dimension (defaults to dim)
            out_dim: Output dimension (defaults to channels)
            dim_mults: Multipliers for dimensions at each level
            embed_class_layers_dims: Dimensions for class embedding MLP
            channels: Number of input/output channels
            self_condition: Whether to use self-conditioning
            resnet_block_groups: Number of groups for ResNet blocks
            learned_variance: Whether to learn the variance
            learned_sinusoidal_cond: Whether to use learned sinusoidal conditioning
            random_fourier_features: Whether to use random Fourier features
            learned_sinusoidal_dim: Dimension for learned sinusoidal embeddings
            sinusoidal_pos_emb_theta: Theta parameter for sinusoidal embeddings
            attn_dim_head: Dimension per attention head
            attn_heads: Number of attention heads
        """
        super().__init__()

        # classifier free guidance stuff
        self.cond_drop_prob = cond_drop_prob
        self.mask_val = mask_val

        # determine dimensions
        self.dim = dim
        self.dim_mults = dim_mults
        self.embed_class_layers_dims = embed_class_layers_dims

        self.seq_length = seq_length

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, self.dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Partial function is used to create a "partial" version of a function. You can think of it as pre-setting some arguments of a function.
        # Here, the ResnetBlock class is being partially set with the groups argument pre-defined. Any subsequent calls to block_klass will be the same as calling ResnetBlock with groups=resnet_block_groups.
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = self.dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(self.dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = self.dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Class embeddings ##, define the conditional embedding 
        # embedded_classes_dim = self.dim * 4
        # self.classes_mlp = nn.Sequential(
        #     nn.Linear(class_dim, embedded_classes_dim),
        #     nn.GELU(),
        #     nn.Linear(embedded_classes_dim, embedded_classes_dim)
        # )

        self.classes_mlp = self.build_classes_mlp(class_dim, self.embed_class_layers_dims)
        embedded_classes_dim = self.embed_class_layers_dims[-1]

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # Handle the case where seq_length = 1, no upsampling and downsampling, only conv1d
            if self.seq_length > 1:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                ]))
            else:
                # If sequence length is already 1, avoid downsampling and just use convolutional layers
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    # nn.Conv1d(dim_in, dim_out, 3, padding=1)
                    Downsample(dim_in, dim_out, 2) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            is_second_last = ind == (len(in_out) - 2)
            is_third_last = ind == (len(in_out) - 3)

            if self.seq_length > 1:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    # Since the original data size not necessarily can be divided by two,
                    #  so if is the second last year of upsample, we need make sure it upsamples to the original size
                    # Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    # Jannik: Added first upsample to half of the sequence length for input of dimension 66 => 33 => 16 => 33 => 66
                    Final_upsample_to_target_length(dim_out, dim_in,
                                                    target_length=int(self.seq_length/2)) if is_third_last else
                    (Final_upsample_to_target_length(dim_out, dim_in,
                                                    target_length=self.seq_length) if is_second_last else
                    (nn.Conv1d(dim_out, dim_in, 3, padding=1) if is_last else
                     Upsample(dim_out, dim_in)))

                ]))
            else:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    # nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    (nn.Conv1d(dim_out, dim_in, 3, padding=1) if is_last else
                     Upsample(dim_out, dim_in, 1))
                ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(self.dim * 2, self.dim, time_emb_dim=time_dim,
                                           classes_emb_dim=embedded_classes_dim)
        self.final_conv = nn.Conv1d(self.dim, self.out_dim, 1)

    def build_classes_mlp(self, class_dim, embed_class_layers_dims):
        """
        Build a multi-layer perceptron (MLP) for class embeddings.
        
        This function constructs a neural network with linear layers and GELU activations
        to transform class embeddings from the input dimension to the specified output dimensions.
        The final activation layer is removed to return raw logits.
        
        Args:
            class_dim (int): Input dimension of the class embeddings
            embed_class_layers_dims (list): List of output dimensions for each layer
            
        Returns:
            nn.Sequential: A sequential neural network for class embedding transformation
        """
        layers = []
        input_dim = copy.copy(class_dim)
        
        # Build layers with linear transformations and GELU activations
        for output_dim in embed_class_layers_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            input_dim = output_dim

        # Remove the last GELU activation to return raw logits
        layers.pop()

        return nn.Sequential(*layers)

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            rescaled_phi=0.,
            **kwargs
    ):
        """
        Forward pass with classifier-free guidance scaling.
        
        This method implements classifier-free guidance by computing both conditional and unconditional
        outputs, then combining them using a scaling factor. It also supports rescaling based on
        standard deviation to maintain statistical properties.
        
        Args:
            *args: Positional arguments passed to the forward method
            cond_scale (float): Scaling factor for classifier-free guidance. 
                               When cond_scale=1, returns fully conditional output.
                               When cond_scale>1, increases the influence of conditioning.
            rescaled_phi (float): Weight for rescaling based on standard deviation.
                                 When rescaled_phi=0, no rescaling is applied.
                                 When rescaled_phi=1, fully uses rescaled output.
            **kwargs: Keyword arguments passed to the forward method
            
        Returns:
            torch.Tensor: Scaled and potentially rescaled logits
        """
        # Get fully conditional output (no dropout applied to conditioning)
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        # If cond_scale is 1, return the conditional output directly
        if cond_scale == 1:
            return logits

        # Get unconditional output (full dropout applied to conditioning)
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)

        # Apply classifier-free guidance scaling:
        # scaled_logits = unconditional + scale * (conditional - unconditional)
        # This is equivalent to: scale * conditional + (1 - scale) * unconditional
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        # If no rescaling is requested, return the scaled logits
        if rescaled_phi == 0.:
            return scaled_logits

        # Apply rescaling based on standard deviation to maintain statistical properties
        # This helps preserve the variance of the original conditional output
        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / (std_fn(scaled_logits) + 1e-6))
        
        # Safety check: if rescaling produces NaN values, fall back to scaled logits
        if torch.isnan(rescaled_logits).any():
            return scaled_logits

        # Combine rescaled and original scaled logits using rescaled_phi weight
        # If rescaled_phi = 1.0, fully use rescaled output
        # If rescaled_phi = 0.0, fully use original scaled output
        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(self, x, time, classes, cond_drop_prob=None):
        """
        Forward pass through the U-Net diffusion model with conditional guidance.
        
        This method implements the main forward pass of the diffusion model, processing
        the input through a U-Net architecture with time embeddings and conditional
        class information. It supports classifier-free guidance through conditional dropout.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
            time (torch.Tensor): Time step tensor of shape (batch_size,)
            classes (torch.Tensor): Class labels tensor of shape (batch_size, num_classes)
            cond_drop_prob (float, optional): Conditional dropout probability for classifier-free guidance.
                                            If None, uses the default value from model initialization.
        
        Returns:
            torch.Tensor: Predicted noise or denoised output of same shape as input
        """
        # Extract batch size and device from input tensor
        batch, device = x.shape[0], x.device
        
        # Use default conditional dropout probability if not provided
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # =============================================================================
        # CONDITIONAL GUIDANCE SETUP
        # =============================================================================
        
        # Apply conditional dropout for classifier-free guidance
        if cond_drop_prob > 0:
            # Create mask to randomly drop conditioning information
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            
            # Apply mask to class embeddings: keep original classes or use mask value
            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes,
                torch.tensor(self.mask_val, device=classes.device)  # Use mask value for dropped conditioning
            )
            
            # Embed class information into conditional variable c
            c = self.classes_mlp(classes_emb)
        else:
            # No dropout: use all class information
            c = self.classes_mlp(classes)

        # =============================================================================
        # U-NET ENCODING PATH (DOWNWARD)
        # =============================================================================
        
        # Initial convolution layer
        x = self.init_conv(x)
        r = x.clone()  # Store residual connection for final skip connection
        
        # Generate time embeddings
        t = self.time_mlp(time)
        
        # Store intermediate features for skip connections
        h = []
        
        # Process through downsampling blocks
        for block1, block2, attn, downsample in self.downs:
            # First residual block with time and conditional embeddings
            x = block1(x, t, c)
            h.append(x)  # Store for skip connection
            
            # Second residual block with time and conditional embeddings
            x = block2(x, t, c)
            x = attn(x)  # Apply attention mechanism
            h.append(x)  # Store for skip connection
            
            # Downsample feature maps
            x = downsample(x)

        # =============================================================================
        # U-NET BOTTLENECK (MIDDLE)
        # =============================================================================
        
        # Middle residual blocks with attention
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        # =============================================================================
        # U-NET DECODING PATH (UPWARD)
        # =============================================================================
        
        # Process through upsampling blocks
        for block1, block2, attn, upsample in self.ups:
            # Concatenate with skip connection from encoding path
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)  # First residual block
            
            # Concatenate with another skip connection
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)  # Second residual block
            x = attn(x)          # Apply attention mechanism
            x = upsample(x)      # Upsample feature maps
        
        # =============================================================================
        # FINAL OUTPUT
        # =============================================================================
        
        # Add final residual connection from input
        x = torch.cat((x, r), dim=1)
        
        # Final residual block and output convolution
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)



# =============================================================================
# DIFFUSION UTILITY FUNCTIONS
# =============================================================================

def extract(a, t, x_shape):
    """
    Extract values from tensor 'a' at indices 't' and reshape to match 'x_shape'.
    
    This function is used to extract time-dependent parameters (like alphas, betas)
    from their respective tensors and reshape them to match the input tensor shape.
    
    Args:
        a: Source tensor to extract values from
        t: Time indices tensor
        x_shape: Target shape for the output tensor
        
    Returns:
        Tensor: Extracted values reshaped to match x_shape
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    Create a linear beta schedule for the diffusion process.
    
    This function generates a linearly increasing sequence of noise levels
    (betas) that control the diffusion process from clean data to pure noise.
    
    Args:
        timesteps: Number of diffusion timesteps
        
    Returns:
        Tensor: Linear beta schedule with shape (timesteps,)
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Create a cosine beta schedule for the diffusion process.
    
    This function generates a cosine-based sequence of noise levels that
    provides better performance than linear schedules. Based on the paper
    "Improved Denoising Diffusion Probabilistic Models".
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small constant to prevent division by zero (default: 0.008)
        
    Returns:
        Tensor: Cosine beta schedule with shape (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# =============================================================================
# GAUSSIAN DIFFUSION MODEL
# =============================================================================

class GaussianDiffusion1D(nn.Module):
    """
    1D Gaussian Diffusion Model with classifier-free guidance and constraint handling.
    
    This class implements a 1D diffusion model that can generate sequences
    conditioned on class embeddings. It supports various objectives (noise prediction,
    x0 prediction, v-parameterization) and includes constraint violation handling
    for physics-informed generation.
    """
    
    def __init__(
            self,
            model,
            *,
            seq_length,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='cosine',
            ddim_sampling_eta=0.,
            auto_normalize=True,
            constraint_violation_weight=0.001,
            constraint_condscale=6.,
            max_sample_step_with_constraint_loss=500,
            constraint_loss_type="one_over_t",
            task_type="car",
            constraint_gt_sample_num=1,
            normalize_xt_by_mean_sigma="False",
    ):
        """
        Initialize the Gaussian Diffusion model.
        
        Args:
            model: U-Net model for denoising
            seq_length: Length of sequences to generate
            timesteps: Number of diffusion timesteps (default: 1000)
            sampling_timesteps: Number of timesteps for sampling (default: timesteps)
            objective: Prediction objective ('pred_noise', 'pred_x0', 'pred_v')
            beta_schedule: Beta schedule type ('linear' or 'cosine')
            ddim_sampling_eta: DDIM sampling parameter (default: 0.0)
            auto_normalize: Whether to auto-normalize inputs/outputs
            constraint_violation_weight: Weight for constraint violation loss
            constraint_condscale: Conditional scaling for constraints
            max_sample_step_with_constraint_loss: Max steps with constraint loss
            constraint_loss_type: Type of constraint loss scaling
            task_type: Type of task (e.g., "car")
            constraint_gt_sample_num: Number of ground truth samples for constraints
            normalize_xt_by_mean_sigma: Whether to normalize by mean/sigma
        """
        super().__init__()
        
        # Model and basic parameters
        self.model = model
        self.channels = self.model.channels
        self.seq_length = seq_length
        self.timesteps = timesteps
        self.objective = objective
        
        # Constraint-related parameters
        self.constraint_violation_weight = constraint_violation_weight
        self.constraint_condscale = constraint_condscale
        self.max_sample_step_with_constraint_loss = max_sample_step_with_constraint_loss
        self.constraint_loss_type = constraint_loss_type
        self.task_type = task_type
        self.constraint_gt_sample_num = constraint_gt_sample_num
        self.normalize_xt_by_mean_sigma = normalize_xt_by_mean_sigma

        # Validate objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # Define beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # Calculate derived parameters
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # Sampling parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # Helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # Register basic diffusion parameters
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # Above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Calculate loss weight based on signal-to-noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # Normalization functions
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise.
        
        Args:
            x_t: Noisy tensor at timestep t
            t: Timestep indices
            noise: Predicted noise
            
        Returns:
            Tensor: Predicted x_0
        """
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        Predict noise from x_t and x_0.
        
        Args:
            x_t: Noisy tensor at timestep t
            t: Timestep indices
            x0: Clean tensor x_0
            
        Returns:
            Tensor: Predicted noise
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        """
        Predict v-parameterization from x_0 and noise.
        
        Args:
            x_start: Clean tensor x_0
            t: Timestep indices
            noise: Noise tensor
            
        Returns:
            Tensor: Predicted v-parameterization
        """
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """
        Predict x_0 from x_t and v-parameterization.
        
        Args:
            x_t: Noisy tensor at timestep t
            t: Timestep indices
            v: V-parameterization tensor
            
        Returns:
            Tensor: Predicted x_0
        """
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Calculate the posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start: Clean tensor x_0
            x_t: Noisy tensor at timestep t
            t: Timestep indices
            
        Returns:
            tuple: (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale=6., rescaled_phi=0.7, clip_x_start=False,
                          rederive_pred_noise=False):
        """
        Get model predictions for noise and x_0.
        
        Args:
            x: Input tensor
            t: Timestep indices
            classes: Class embeddings
            cond_scale: Conditional scaling factor
            rescaled_phi: Rescaling parameter
            clip_x_start: Whether to clip predicted x_0
            rederive_pred_noise: Whether to rederive noise after clipping
            
        Returns:
            ModelPrediction: Named tuple containing (pred_noise, pred_x_start)
        """
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale=cond_scale,
                                                          rescaled_phi=rescaled_phi)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised=True):
        """
        Calculate the mean and variance of p(x_{t-1} | x_t).
        
        Args:
            x: Input tensor
            t: Timestep indices
            classes: Class embeddings
            cond_scale: Conditional scaling factor
            rescaled_phi: Rescaling parameter
            clip_denoised: Whether to clip denoised values
            
        Returns:
            tuple: (mean, variance, log_variance)
        """
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # Obtain mean from model prediction, obtain var from posterior?
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale=6., rescaled_phi=0.7,
                 clip_denoised=True):
        """
        Sample x_{t-1} from x_t using the reverse diffusion process.
        
        This method implements the reverse diffusion step, sampling x_{t-1} from x_t
        using the learned model predictions and the posterior distribution.
        
        Args:
            x: Current noisy tensor x_t
            t: Current timestep (can be int or tensor)
            classes: Class embeddings for conditioning
            cond_scale: Conditional scaling factor for classifier-free guidance
            rescaled_phi: Rescaling parameter
            clip_denoised: Whether to clip denoised values
            
        Returns:
            tuple: (predicted x_{t-1}, predicted x_0)
        """
        b, *_, device = *x.shape, x.device
        
        # Handle different types of timestep input
        if isinstance(t, int):
            batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        else:  # multiple different t
            batched_times = t.clone().to(device=x.device, dtype=torch.long)
        # Get mean and var to sample x_{t-1}. Mean is a function of x_t, var is from posterior
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, classes=classes,
                                                                          cond_scale=cond_scale,
                                                                          rescaled_phi=rescaled_phi,
                                                                          clip_denoised=clip_denoised)
        if isinstance(t, int):
            noise = torch.randn_like(x) if (t > 0) else 0.
        else:
            noise = torch.randn_like(x) if (len(t.shape) >= 1 or t > 0) else 0.
        
        # Sample using mean and variance (DDPM algorithm 2)
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale=6., rescaled_phi=0.7):
        """
        Generate samples using the full reverse diffusion process.
        
        This method implements the complete sampling loop, starting from pure noise
        and gradually denoising to generate samples conditioned on class embeddings.
        
        Args:
            classes: Class embeddings for conditioning
            shape: Shape of the samples to generate (batch_size, channels, seq_length)
            cond_scale: Conditional scaling factor for classifier-free guidance
            rescaled_phi: Rescaling parameter
            
        Returns:
            Tensor: Generated samples
        """
        batch, device = shape[0], self.betas.device

        # Default image to be randomly sampled from [0, 1]?
        img = torch.randn(shape, device=device)

        x_start = None

        # The giant sample loop, loop over each time steps, and sample the x_(t-1) at each steps
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # print(t)
            # print(torch.max(img))
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale=6., rescaled_phi=0.7, clip_denoised=True):
        """
        Generate samples using DDIM (Denoising Diffusion Implicit Models) sampling.
        
        DDIM is a deterministic sampling method that can generate high-quality
        samples with fewer timesteps than the standard DDPM sampling.
        
        Args:
            classes: Class embeddings for conditioning
            shape: Shape of the samples to generate
            cond_scale: Conditional scaling factor for classifier-free guidance
            rescaled_phi: Rescaling parameter
            clip_denoised: Whether to clip denoised values
            
        Returns:
            Tensor: Generated samples
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale=cond_scale,
                                                             clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale=6., rescaled_phi=0.7):
        """
        Generate samples using the diffusion model with classifier-free guidance.
        
        This method is the main entry point for sampling from the diffusion model.
        It automatically chooses between DDPM and DDIM sampling based on the
        model configuration.
        
        Args:
            classes: Class embeddings for conditioning the generation
            cond_scale: Conditional scaling factor for classifier-free guidance (default: 6.0)
            rescaled_phi: Rescaling parameter for the diffusion process (default: 0.7)
            
        Returns:
            Tensor: Generated samples with shape (batch_size, channels, seq_length)
        """
        # Extract dimensions from the input classes tensor
        batch_size, seq_length, channels = classes.shape[0], self.seq_length, self.channels
        
        # Choose sampling function based on whether DDIM sampling is enabled
        # DDIM is faster but DDPM is the original sampling method
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        # Generate samples using the selected sampling function
        return sample_fn(classes, (batch_size, channels, seq_length), cond_scale, rescaled_phi)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        """
        Interpolate between two samples.
        
        This method performs spherical linear interpolation (SLERP) between
        two samples at a given timestep.
        
        Args:
            x1: First sample
            x2: Second sample
            t: Timestep for interpolation (default: None, uses random timestep)
            lam: Interpolation parameter (0 = x1, 1 = x2)
            
        Returns:
            Tensor: Interpolated sample
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0) - the forward diffusion process.
        
        This method adds noise to x_0 according to the diffusion schedule
        to get x_t at timestep t.
        
        Args:
            x_start: Clean tensor x_0
            t: Timestep indices
            noise: Optional noise tensor (if None, random noise is generated)
            
        Returns:
            Tensor: Noisy tensor x_t
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @autocast(enabled=False)
    def q_sample_many(self, x_start, t, sample_num, noise=None):
        """
        Sample multiple times from q(x_t | x_0) - the forward diffusion process.
        
        This method generates multiple noisy samples from the same clean input x_0
        by adding different noise realizations according to the diffusion schedule.
        
        Args:
            x_start: Clean tensor x_0 of shape (batch_size, channel_size, feature_size)
            t: Timestep indices
            sample_num: Number of different noise samples to generate
            noise: Optional noise tensor of shape (batch_size, channel_size, feature_size, sample_num)
                   (if None, random noise is generated)
            
        Returns:
            Tensor: Multiple noisy samples of shape (batch_size, channel_size, feature_size, sample_num)
        """
        # Extract dimensions from input tensor
        batch_size, channel_size, feature_size = x_start.shape

        # Handle edge case: replace -1 timesteps with 0 (no noise added)
        t = torch.where(t == -1, torch.zeros_like(t), t)

        # Generate random noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, channel_size, feature_size, sample_num, device=x_start.device)

        # Initialize a list to collect sampled tensors
        x_ts_samples = []
        
        # Generate multiple samples by adding different noise realizations
        for i in range(sample_num):  # Iterate over the sample_num noise samples
            # Extract the i-th noise sample for all tensors in the batch
            noise_i = noise[:, :, :, i]
            
            # Apply forward diffusion process: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
            x_t_i = (
                    extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                    extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise_i
            )
            
            # Add sample dimension and append to list
            # Each element in x_ts_samples will have shape (batch_size, channel_size, feature_size, 1)
            x_ts_samples.append(x_t_i.unsqueeze(-1))

        # Concatenate all samples along the last dimension to create final tensor
        # Result shape: (batch_size, channel_size, feature_size, sample_num)
        x_ts = torch.cat(x_ts_samples, dim=-1)
        return x_ts

    def p_losses(self, x_start, t, *, classes, noise=None):
        """
        Compute the training loss for the diffusion model with physics-informed constraints.
        
        This method computes the main diffusion loss and optionally includes constraint violation
        loss for physics-informed training. The constraint loss can be computed using different
        strategies including ground truth sampling and various statistical measures.
        
        Args:
            x_start: Clean tensor x_0 of shape (batch_size, channels, seq_length)
            t: Timestep indices of shape (batch_size,)
            classes: Class embeddings for conditioning of shape (batch_size, class_dim)
            noise: Optional noise tensor (if None, random noise is generated)
            
        Returns:
            Tensor: Total loss value combining diffusion and constraint losses
        """
        # =============================================================================
        # STEP 1: BASIC DIFFUSION SETUP
        # =============================================================================
        
        # Generate random noise if not provided for the forward diffusion process
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Forward diffusion: add noise to x_0 to get x_t according to diffusion schedule
        # x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Get model prediction for the noisy input x_t
        model_out = self.model(x_t, t, classes)

        # Determine the target for the loss based on the objective
        if self.objective == 'pred_noise':
            target = noise  # Model predicts the noise that was added
        elif self.objective == 'pred_x0':
            target = x_start  # Model predicts the clean input
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)  # Model predicts v-parameterization
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # =============================================================================
        # STEP 2: CONSTRAINT VIOLATION COMPUTATION
        # =============================================================================
        
        # Check if DDIM sampling is being used (not supported for constraint loss)
        if self.is_ddim_sampling:
            print("don't use ddim sampling!")
            exit()
        else:
            # Sample x_{t-1} using classifier-free guidance for constraint evaluation
            # This gives us a denoised sample to evaluate physics constraints
            cond_scale = self.constraint_condscale
            rescaled_phi = 0.7
            x_t_1, _ = self.p_sample(x_t, t, classes, cond_scale, rescaled_phi)

        # =============================================================================
        # STEP 3: ANALYTICAL NORMALIZATION
        # =============================================================================
        
        # Compute analytical mean and standard deviation for x_{t-1} based on diffusion schedule
        # This provides theoretical bounds for the denoised samples
        safe_t_1 = torch.where((t - 1) == -1, torch.zeros_like(t), t - 1)  # Handle t=0 case
        x_t_1_analytical_mean = extract(self.sqrt_alphas_cumprod, safe_t_1, x_start.shape) * x_start
        x_t_1_analytical_sigma = extract(self.sqrt_one_minus_alphas_cumprod, safe_t_1, x_start.shape)

        # Compute 3-sigma bounds (99.7% data coverage under normal distribution)
        x_t_1_analytical_lower_bound = x_t_1_analytical_mean - 3 * x_t_1_analytical_sigma
        x_t_1_analytical_upper_bound = x_t_1_analytical_mean + 3 * x_t_1_analytical_sigma

        # Normalize x_t_1 to [0,1] range for constraint evaluation
        if self.normalize_xt_by_mean_sigma == "True":
            # Use analytical bounds for normalization
            x_t_1 = (x_t_1 - x_t_1_analytical_lower_bound) / (
                        x_t_1_analytical_upper_bound - x_t_1_analytical_lower_bound)
            x_t_1 = torch.clamp(x_t_1, min=0.0, max=1.0)
        else:
            # Use standard [-1,1] to [0,1] normalization
            x_t_1 = torch.clamp(x_t_1, min=-1.0, max=1.0)
            x_t_1 = (x_t_1 + 1.0) / 2.0

        # =============================================================================
        # STEP 4: CONSTRAINT FUNCTION SELECTION
        # =============================================================================
        
        # Choose the appropriate constraint violation function based on task type
        if self.task_type == "car":
            # Car dynamics constraints (e.g., velocity, acceleration limits)
            from denoising_diffusion_pytorch.constraint_violation_function_improved_car import \
                get_constraint_violation_car
            get_constraint_function = get_constraint_violation_car
        elif self.task_type == "tabletop":
            # Tabletop manipulation constraints (e.g., object stability)
            from denoising_diffusion_pytorch.constraint_violation_function_improved_tabletop_setupv2 import \
                get_constraint_violation_tabletop
            get_constraint_function = get_constraint_violation_tabletop
        elif self.task_type == "cr3bp":
            # Circular Restricted Three-Body Problem constraints
            # CR3BP constraint function would be implemented here
            pass
        else:
            print("wrong task type")
            exit()

        # =============================================================================
        # STEP 5: CONSTRAINT LOSS COMPUTATION
        # =============================================================================
        
        # Compute constraint violation loss based on the specified loss type
        if self.constraint_loss_type == "NA":
            # No constraint loss - return standard diffusion loss only
            loss = F.mse_loss(model_out, target, reduction='none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss * extract(self.loss_weight, t, loss.shape)
            return loss.mean()
            
        elif self.constraint_loss_type == "one_over_t":
            # Simple constraint loss with 1/(t+1) scaling
            # This reduces constraint influence at later timesteps
            nn_violation_loss = get_constraint_function(x_t_1.view(x_start.shape[0], -1),
                                                        classes,  # Class labels for conditioning
                                                        1. / (t + 1),  # Time-dependent scaling
                                                        x_start.device)
            violation_loss_final_use = nn_violation_loss

        else:
            # Advanced constraint loss types using ground truth sampling
            # These methods compare model predictions against ground truth statistics
            
            # =============================================================================
            # STEP 5A: GROUND TRUTH SAMPLING
            # =============================================================================
            
            # Generate multiple ground truth samples from the same x_0 at timestep t-1
            # This provides a statistical baseline for constraint evaluation
            # x_t_1_gt has shape (batch_size, channel_size, feature_size, sample_size)
            x_t_1_gt = self.q_sample_many(x_start=x_start, t=t - 1, sample_num=self.constraint_gt_sample_num)

            # Normalize ground truth samples using the same method as x_t_1
            if self.normalize_xt_by_mean_sigma == "True":
                # Expand bounds to match sample dimension
                expanded_lower_bound = x_t_1_analytical_lower_bound.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.constraint_gt_sample_num)
                expanded_upper_bound = x_t_1_analytical_upper_bound.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.constraint_gt_sample_num)

                x_t_1_gt = (x_t_1_gt - expanded_lower_bound) / (expanded_upper_bound - expanded_lower_bound)
                x_t_1_gt = torch.clamp(x_t_1_gt, min=0.0, max=1.0)
            else:
                x_t_1_gt = torch.clamp(x_t_1_gt, min=-1.0, max=1.0)
                x_t_1_gt = (x_t_1_gt + 1.0) / 2.0

            # Extract dimensions for reshaping
            batch_size, channel_size, feature_size, sample_size = x_t_1_gt.shape

            # =============================================================================
            # STEP 5B: DATA RESHAPING FOR CONSTRAINT EVALUATION
            # =============================================================================
            
            # Reshape ground truth samples for batch processing
            # Convert from (batch, channel, feature, sample) to (batch*sample, feature)
            # This allows processing all samples together
            reshaped_x_t_1_gt = x_t_1_gt.permute(0, 3, 1, 2).reshape(-1, feature_size)

            # Repeat class labels to match the expanded batch size
            # Each original sample gets repeated for each ground truth sample
            expanded_classes = classes.repeat_interleave(self.constraint_gt_sample_num, dim=0)

            # Compute constraint violations for all ground truth samples
            violation_losses = get_constraint_function(reshaped_x_t_1_gt,
                                                       expanded_classes,  # Repeated class labels
                                                       1.,  # Constant scaling factor
                                                       x_start.device)

            # =============================================================================
            # STEP 5C: STATISTICAL COMPUTATION
            # =============================================================================
            
            # Reshape violation losses back to separate batch and sample dimensions
            # Shape: (batch_size, sample_size) - each row contains violations for one batch item
            reshaped_violation_losses = violation_losses.view(-1, self.constraint_gt_sample_num)

            # Compute statistics across ground truth samples for each batch item
            gt_average_violation_loss = reshaped_violation_losses.mean(dim=1)  # Mean violation
            gt_std_loss = reshaped_violation_losses.std(dim=1)  # Standard deviation of violations

            # Compute constraint violation for the model's prediction
            nn_violation_loss = get_constraint_function(x_t_1.view(x_start.shape[0], -1),
                                                       classes,  # Original class labels
                                                       1.,  # Constant scaling factor
                                                       x_start.device)

            # =============================================================================
            # STEP 5D: ADVANCED LOSS COMPUTATION STRATEGIES
            # =============================================================================
            
            # Choose the specific constraint loss computation method
            if self.constraint_loss_type == "gt_threshold":
                # Only penalize when model violation exceeds ground truth average
                difference = nn_violation_loss - gt_average_violation_loss
                violation_loss_final_use = torch.max(difference, torch.zeros_like(difference))

            elif self.constraint_loss_type == "gt_scaled":
                # Scale model violation by ground truth average (relative comparison)
                violation_loss_final_use = nn_violation_loss / gt_average_violation_loss

            elif self.constraint_loss_type == "gt_std":
                # Z-score: how many standard deviations away from ground truth mean
                violation_loss_final_use = (nn_violation_loss - gt_average_violation_loss) / gt_std_loss

            elif self.constraint_loss_type == "gt_std_absolute":
                # Absolute Z-score: magnitude of deviation from ground truth
                violation_loss_final_use = torch.abs(nn_violation_loss - gt_average_violation_loss) / gt_std_loss

            elif self.constraint_loss_type == "gt_std_threshold":
                # Thresholded Z-score: only penalize when significantly above ground truth
                difference = nn_violation_loss - gt_average_violation_loss
                violation_loss_final_use = torch.max(difference, torch.zeros_like(difference)) / gt_std_loss

            elif self.constraint_loss_type == "gt_log_likelihood":
                # Log-likelihood based on ground truth statistics
                # Assumes violations follow normal distribution around ground truth mean
                violation_loss_final_use = torch.square((nn_violation_loss - gt_average_violation_loss) / gt_std_loss)

            else:
                print("wrong constraint_loss_type")
                exit()

        # =============================================================================
        # STEP 6: TIMESTEP MASKING AND LOSS COMBINATION
        # =============================================================================
        
        # Apply constraint loss only to early timesteps (where constraints matter most)
        # Create a mask that is 1 for t <= max_sample_step_with_constraint_loss, 0 otherwise
        mask = t <= self.max_sample_step_with_constraint_loss
        mask = mask.float()
        
        # Apply the mask to zero out constraint loss for later timesteps
        masked_violation_loss = violation_loss_final_use * mask

        # Compute the mean violation loss across the batch
        violation_loss_final_use_mean = torch.mean(masked_violation_loss)

        # Get the constraint violation weight coefficient
        coef = torch.tensor(self.constraint_violation_weight)

        # =============================================================================
        # STEP 7: FINAL LOSS COMPUTATION
        # =============================================================================
        
        # Compute the main diffusion loss (MSE between model output and target)
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')  # Average across spatial dimensions

        # Apply timestep-dependent loss weighting
        loss = loss * extract(self.loss_weight, t, loss.shape)
        
        # Combine diffusion loss with constraint violation loss
        # Total loss = diffusion_loss + constraint_weight * constraint_violation_loss
        return loss.mean() + coef * violation_loss_final_use_mean

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)

        return self.p_losses(img, t, *args, **kwargs)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer1D(object):
    """
    1D Diffusion Model Trainer with validation and checkpoint management.
    
    This class handles the training loop for the 1D diffusion model,
    including data loading, optimization, validation, and model checkpointing.
    It also integrates with Weights & Biases for experiment tracking.
    """
    
    def __init__(
            self,
            diffusion_model: GaussianDiffusion1D,
            dataset: Dataset,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1.,
            num_workers=1,
            wandb_project_name="diffusion_for_cr3bp_indirect",
            training_data_range="0_1",
            training_data_num=300000,
            training_random_seed=0
    ):
        """
        Initialize the trainer.
        
        Args:
            diffusion_model: The diffusion model to train
            dataset: Training dataset
            train_batch_size: Batch size for training
            gradient_accumulate_every: Gradient accumulation steps
            train_lr: Learning rate
            train_num_steps: Total number of training steps
            ema_update_every: EMA update frequency
            ema_decay: EMA decay rate
            adam_betas: Adam optimizer beta parameters
            save_and_sample_every: Frequency of saving and sampling
            num_samples: Number of samples to generate during evaluation
            results_folder: Folder to save results
            amp: Whether to use automatic mixed precision
            mixed_precision_type: Type of mixed precision
            split_batches: Whether to split batches across devices
            max_grad_norm: Maximum gradient norm for clipping
            num_workers: Number of data loader workers
            wandb_project_name: Weights & Biases project name
            training_data_range: Range of training data
            training_data_num: Number of training samples
            training_random_seed: Random seed for training
        """
        super().__init__()

        # Initialize accelerator for distributed training
        # Handle MPS device configuration
        if torch.backends.mps.is_available():
            # MPS doesn't support mixed precision, so force it to 'no'
            mixed_precision_type = 'no'
        
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # Model and dataset
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # Split dataset into training and validation
        train_length = int(0.9 * len(dataset))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

        # Create data loaders
        train_dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, 
                             pin_memory=True, num_workers=num_workers)
        val_dl = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, 
                           pin_memory=True, num_workers=num_workers)

        train_dl = self.accelerator.prepare(train_dl)
        val_dl = self.accelerator.prepare(val_dl)

        self.train_dl = cycle(train_dl)
        self.val_dl = val_dl

        # Optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # Exponential moving average
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # Results folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.batches_per_epoch = len(dataset) // self.batch_size
        self.train_lr = train_lr

        # Best checkpoints tracking
        self.best_checkpoints = []
        
        # Initialize wandb
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    project=wandb_project_name,
                    config={
                        'train_batch_size': train_batch_size,
                        'train_lr': train_lr,
                        'train_num_steps': train_num_steps,
                        'gradient_accumulate_every': gradient_accumulate_every,
                        'ema_decay': ema_decay,
                        'max_grad_norm': max_grad_norm,
                        'training_data_range': training_data_range,
                        'training_data_num': training_data_num,
                        'training_random_seed': training_random_seed,
                    }
                )
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                print("Continuing without wandb logging...")

    @property
    def device(self):
        """Get the device used by the accelerator."""
        return self.accelerator.device

    def save(self, milestone):
        """
        Save model checkpoint.
        
        Args:
            milestone: Checkpoint identifier
        """
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': '1.0.0'
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        """
        Load model checkpoint.
        
        Args:
            milestone: Checkpoint identifier
        """
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        # Handle EMA state dict modifications
        ema_state_dict = data['ema']
        if 'initted' in data['ema'] and ema_state_dict['initted'].shape == torch.Size([]):
            ema_state_dict['initted'] = ema_state_dict['initted'].unsqueeze(0)
        if 'step' in ema_state_dict and ema_state_dict['step'].shape == torch.Size([]):
            ema_state_dict['step'] = ema_state_dict['step'].unsqueeze(0)

        # Print shapes for debugging
        print("Shape of initted in EMA state dict:", ema_state_dict.get('initted', 'Not found').shape)
        print("Shape of step in EMA state dict:", ema_state_dict.get('step', 'Not found').shape)
        #############################################################################################

        # Load model state
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(ema_state_dict)

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        """
        Main training loop.
        
        This method implements the complete training loop with validation,
        checkpointing, and logging.
        """
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            # Track best validation loss
            best_val_loss = torch.tensor(float("inf"))

            while self.step < self.train_num_steps:

                total_loss = 0.

                # Gradient accumulation loop
                for _ in range(self.gradient_accumulate_every):
                    # Get training batch
                    training_sequence, training_classes = next(self.train_dl)
                    training_sequence, training_classes = training_sequence.to(device), training_classes.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(training_sequence, classes=training_classes)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                # Log training loss
                if self.accelerator.is_main_process:
                    try:
                        wandb.log({'train_loss': total_loss, 'step': self.step})
                    except:
                        pass  # Silently ignore if wandb is not available

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                
                if accelerator.is_main_process:
                    self.ema.update()

                    # Validation and checkpointing at epoch boundaries
                    if self.step % self.batches_per_epoch == 0 and self.step != 0:
                        milestone = self.step // self.batches_per_epoch
                        print(f"Epoch {milestone}")

                        val_loss = self.compute_validation_loss()

                        # Log validation loss
                        if self.accelerator.is_main_process:
                            try:
                                wandb.log({'val_loss': val_loss, 'epoch': milestone})
                            except:
                                pass  # Silently ignore if wandb is not available

                        # Save checkpoint if validation loss improves
                        if val_loss < best_val_loss:
                            self.save(f"epoch-{milestone}")
                            best_val_loss = val_loss
                            self.update_best_checkpoints(val_loss, f"epoch-{milestone}")

                pbar.update(1)

        accelerator.print('training complete')

    def update_best_checkpoints(self, val_loss, milestone):
        """
        Update the list of best checkpoints.
        
        Args:
            val_loss: Validation loss value
            milestone: Checkpoint identifier
        """
        # Add new checkpoint and sort by validation loss
        self.best_checkpoints.append((val_loss, str(self.results_folder / f'model-{milestone}.pt')))
        self.best_checkpoints.sort(key=lambda x: x[0])

        # Keep only top 2 checkpoints
        if len(self.best_checkpoints) > 2:
            _, checkpoint_to_remove = self.best_checkpoints.pop(2)  # Remove the 4th checkpoint
            if os.path.exists(checkpoint_to_remove):
                os.remove(checkpoint_to_remove)  # Delete the checkpoint file

    def compute_validation_loss(self):
        """
        Compute validation loss on the validation set.
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_val_loss = 0.

        for val_seq, val_classes in self.val_dl:
            val_seq, val_classes = val_seq.to(self.device), val_classes.to(self.device)

            with torch.no_grad():
                val_loss = self.model(val_seq, classes=val_classes)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(self.val_dl)
        return average_val_loss
