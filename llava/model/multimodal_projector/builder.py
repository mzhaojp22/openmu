import torch
import torch.nn as nn
import re
from transformers import PretrainedConfig, PerceiverModel


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class PerceiverConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PerceiverModel`]. It is used to instantiate an
    Perceiver model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Perceiver
    [deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_latents (`int`, *optional*, defaults to 256):
            The number of latents.
        d_latents (`int`, *optional*, defaults to 1280):
            Dimension of the latent embeddings.
        d_model (`int`, *optional*, defaults to 768):
            Dimension of the inputs. Should only be provided in case [*PerceiverTextPreprocessor*] is used or no
            preprocessor is provided.
        num_blocks (`int`, *optional*, defaults to 1):
            Number of blocks in the Transformer encoder.
        num_self_attends_per_block (`int`, *optional*, defaults to 26):
            The number of self-attention layers per block.
        num_self_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each self-attention layer in the Transformer encoder.
        num_cross_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each cross-attention layer in the Transformer encoder.
        qk_channels (`int`, *optional*):
            Dimension to project the queries + keys before applying attention in the cross-attention and self-attention
            layers of the encoder. Will default to preserving the dimension of the queries if not specified.
        v_channels (`int`, *optional*):
            Dimension to project the values before applying attention in the cross-attention and self-attention layers
            of the encoder. Will default to preserving the dimension of the queries if not specified.
        cross_attention_shape_for_attention (`str`, *optional*, defaults to `"kv"`):
            Dimension to use when downsampling the queries and keys in the cross-attention layer of the encoder.
        self_attention_widening_factor (`int`, *optional*, defaults to 1):
            Dimension of the feed-forward layer in the cross-attention layer of the Transformer encoder.
        cross_attention_widening_factor (`int`, *optional*, defaults to 1):
            Dimension of the feed-forward layer in the self-attention layers of the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_query_residual (`float`, *optional*, defaults to `True`):
            Whether to add a query residual in the cross-attention layer of the encoder.
        vocab_size (`int`, *optional*, defaults to 262):
            Vocabulary size for the masked language modeling model.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that the masked language modeling model might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).
        image_size (`int`, *optional*, defaults to 56):
            Size of the images after preprocessing, for [`PerceiverForImageClassificationLearned`].
        train_size (`List[int]`, *optional*, defaults to `[368, 496]`):
            Training size of the images for the optical flow model.
        num_frames (`int`, *optional*, defaults to 16):
            Number of video frames used for the multimodal autoencoding model.
        audio_samples_per_frame (`int`, *optional*, defaults to 1920):
            Number of audio samples per frame for the multimodal autoencoding model.
        samples_per_patch (`int`, *optional*, defaults to 16):
            Number of audio samples per patch when preprocessing the audio for the multimodal autoencoding model.
        output_shape (`List[int]`, *optional*, defaults to `[1, 16, 224, 224]`):
            Shape of the output (batch_size, num_frames, height, width) for the video decoder queries of the multimodal
            autoencoding model. This excludes the channel dimension.
        output_num_channels (`int`, *optional*, defaults to 512):
            Number of output channels for each modalitiy decoder.

    Example:

    ```python
    >>> from transformers import PerceiverModel, PerceiverConfig

    >>> # Initializing a Perceiver deepmind/language-perceiver style configuration
    >>> configuration = PerceiverConfig()

    >>> # Initializing a model from the deepmind/language-perceiver style configuration
    >>> model = PerceiverModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "perceiver"

    def __init__(
        self,
        num_latents=256,
        d_latents=1280,
        d_model=768,
        num_blocks=1,
        num_self_attends_per_block=26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
        vocab_size=262,
        max_position_embeddings=2048,
        image_size=56,
        train_size=[368, 496],
        num_frames=16,
        audio_samples_per_frame=1920,
        samples_per_patch=16,
        output_shape=[1, 16, 224, 224],
        output_num_channels=512,
        _label_trainable_num_channels=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_latents = num_latents
        self.d_latents = d_latents
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.cross_attention_shape_for_attention = cross_attention_shape_for_attention
        self.self_attention_widening_factor = self_attention_widening_factor
        self.cross_attention_widening_factor = cross_attention_widening_factor
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_query_residual = use_query_residual
        # masked language modeling attributes
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        # image classification attributes
        self.image_size = image_size
        # flow attributes
        self.train_size = train_size
        # multimodal autoencoding attributes
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
        self.output_shape = output_shape
        self.output_num_channels = output_num_channels
        self._label_trainable_num_channels = _label_trainable_num_channels


class PerceiverWithMLPProject(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super().__init__()
        # self.perceiver_cfg = PerceiverConfig(
        #     num_self_attends_per_block=8, num_latents=64, d_latents=768, d_model=768
        # )
        self.perceiver_cfg = PerceiverConfig(
            num_self_attends_per_block=4, num_latents=64, d_latents=1280, d_model=768
        )
        self.perceiver = PerceiverModel(
            self.perceiver_cfg,
            decoder=None,
            input_preprocessor=None,
            output_postprocessor=None,
        )
        self.mlp1 = nn.Linear(1280, mm_hidden_size)
        # self.mlp1 = nn.Linear(768, mm_hidden_size)
        self.mlp2 = nn.Linear(mm_hidden_size, hidden_size)

    def forward(self, x):
        # batch_size, seq_length, _ = x.size() # bs x 512 x 768
        out = self.perceiver(x, output_hidden_states=True, return_dict=True)
        out = out.last_hidden_state  # bs x num_latents x d_latents
        out = self.mlp1(out)
        out = self.mlp2(out)  # bs x num_latents x hidden_size
        return out


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "perceiver":
        return PerceiverWithMLPProject(config.mm_hidden_size, config.hidden_size)

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
