import torch
from torch import nn
import numpy as np
import torchaudio
from .audio_mae.models_vit import vit_base_patch16 as finetunedmae_vit_base_patch16
from timm.models.layers import to_2tuple
from scipy.signal import resample_poly


def load_resample_audio(file_path, target_sr):
    x, sr = torchaudio.load(file_path, normalize=True, channels_first=True)
    if x.ndim > 1:
        x = x.mean(dim=0, keepdim=False)
    x = x.numpy()
    if sr != target_sr:
        x = np.float32(resample_poly(x, target_sr, sr))
    return np.float32(x), sr, target_sr


class PatchEmbed_new_llava(nn.Module):
    """Flexible Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )  # with overlapped patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    @torch.no_grad
    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


TARGET_LENGTH = {"audioset": 1024}
AUDIO_CONF = {
    "num_mel_bins": 128,
    "freqm": 0,  # 0 for pretrain
    "timem": 0,  # 0 for pretrain
    "ft_freqm": 0,  # 48 for finetune; NOTE: ALTERTED TO ZERO INTRODUCING RANDOMNESS
    "ft_timem": 0,  # 192 for finetune; NOTE:  ALTERTED TO ZERO INTRODUCING RANDOMNESS
    "mixup": 0,  # pretrain
    "dataset": "audioset",
    "mode": "train",
    "mean": -4.2677393,
    "std": 4.5689974,
    "noise": False,
}


class AudioPreprocessor:
    """Handling the preprocessing of both 10s and 30s audios"""

    def __init__(self, freqm, timem, target_length) -> None:
        self.audio_conf = AUDIO_CONF
        self.freqm = freqm
        self.timem = timem
        self.target_length = target_length

    def _wav2fbank(self, filename):
        waveform, _, target_sr = load_resample_audio(filename, 16000)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            torch.tensor(waveform).unsqueeze(0),
            htk_compat=True,
            sample_frequency=target_sr,  # converting wrt target_sr
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.audio_conf.get("num_mel_bins"),
            dither=0.0,
            frame_shift=10,
        )
        n_frames = fbank.shape[0]
        # for 10s audios, just zero padding
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[: self.target_length, :]
        return fbank, 0

    def preprocess(self, datum, *args, **kwargs):
        fbank, _ = self._wav2fbank(datum["local_audio_path"])
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0, 1).unsqueeze(0)  # 1, 128, 1024 (..., freq, time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)  # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1)  # time, freq
        fbank = (fbank - self.audio_conf["mean"]) / (self.audio_conf["std"] * 2)
        # [time_frame_num, frequency_bins], e.g., [1024*3, 128]
        return fbank.unsqueeze(0)


class AudioMAEencoder(nn.Module):
    """audio_tower_cfg: ModelArgs"""

    def __init__(self, audio_tower_cfg, delay_load=False) -> None:
        super().__init__()
        self.audio_num_pooling_tokens = audio_tower_cfg.audio_num_pooling_tokens
        if "vitb_pretrained.pth" in audio_tower_cfg.audio_pretrained_ckpt_path:
            raise
        elif "vitb_finetuned.pth" in audio_tower_cfg.audio_pretrained_ckpt_path:
            # audio_mae.models_vit.VisionTransformer
            mae = finetunedmae_vit_base_patch16(
                num_classes=527,  # default from main_finetune_as.py of audiomae
                drop_path_rate=0.1,
                global_pool=True,
                mask_2d=True,
                use_custom_patch=False,
            )
            # AudioMAE can only handle 512 patches due to position embedding
            # so here /3 when defining and loading the model
            # bagging behaviour adapt in forward
            # audio_tower_cfg.audio_input_target_length 3072
            mae.patch_embed = PatchEmbed_new_llava(
                img_size=(audio_tower_cfg.audio_input_target_length // 3, 128),
                patch_size=(16, 16),
                in_chans=1,
                embed_dim=768,
                stride=16,
            )  # no overlap. stride=img_size=16
            num_patches = mae.patch_embed.num_patches  # 512
            mae.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, 768), requires_grad=False
            )  # fixed sin-cos embedding
            self.mae = mae
            freqm = AUDIO_CONF["ft_freqm"]
            timem = AUDIO_CONF["ft_timem"]
        else:
            raise ValueError("Audio Tower type not supported.")

        self.image_processor = AudioPreprocessor(
            freqm=freqm,
            timem=timem,
            target_length=audio_tower_cfg.audio_input_target_length,  # 512*3
        )
        self.hidden_size = 768
        self.is_loaded = False
        if not delay_load:
            print(
                f"\n=>Loading pretrained audio ckpt: {audio_tower_cfg.audio_pretrained_ckpt_path}"
            )
            ckpt = torch.load(
                audio_tower_cfg.audio_pretrained_ckpt_path, map_location="cpu"
            )["model"]
            msg = self.mae.load_state_dict(ckpt, strict=True)
            self.is_loaded = True
            print(f"\n=>Loaded pretrained audio ckpt. MSG: {msg}")
        self.mae.requires_grad_(False)

    @torch.no_grad()
    def pool_freq(self, x):
        # x: bs * 3, 512, h
        bs3, h = x.shape[0], x.shape[2]
        # x = x.view(bs3, -1, 64, h).mean(dim=1)  # pooling 8 patches, bs*3 x 64 x 768
        assert self.audio_num_pooling_tokens <= 512
        x = x.view(bs3, -1, 512 // self.audio_num_pooling_tokens, h).mean(dim=1)
        return x

    @torch.no_grad()
    def forward(self, x):
        # x: [16, 1, 3072, 128])
        bs, num_chan, num_frames, num_bins = x.shape
        assert num_frames == self.image_processor.target_length
        xxx = torch.cat(torch.split(x, num_frames // 3, dim=2), dim=0)
        all_hidden = self.mae.forward_features_no_pooling(xxx)[:, 1:, :]
        all_hidden = self.pool_freq(all_hidden)  # bs*3 x 64 x hidden
        x = torch.cat(torch.split(all_hidden, bs, dim=0), dim=1)
        return x
