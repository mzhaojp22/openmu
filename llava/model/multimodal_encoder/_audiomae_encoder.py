import torch
from torch import nn
import numpy as np
import torchaudio
from .audio_mae.models_mae import mae_vit_base_patch16_dec512d8b
from .audio_mae.models_vit import vit_base_patch16 as finetunedmae_vit_base_patch16
from timm.models.layers import to_2tuple
from scipy.signal import resample_poly


# def load_resample_audio(file_path, target_sr):
#     x, sr = torchaudio.load(file_path, normalize=True, channels_first=True)
#     if x.ndim > 1:
#         x = x.mean(dim=0, keepdim=False)
#     x = x.numpy()
#     if sr != target_sr:
#         x = np.float32(resample_poly(x, target_sr, sr))
#     return np.float32(x), sr, target_sr


# class PatchEmbed_new_llava(nn.Module):
#     """Flexible Image to Patch Embedding"""

#     def __init__(
#         self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10
#     ):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         stride = to_2tuple(stride)

#         self.img_size = img_size
#         self.patch_size = patch_size

#         self.proj = nn.Conv2d(
#             in_chans, embed_dim, kernel_size=patch_size, stride=stride
#         )  # with overlapped patches
#         # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#         # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
#         # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
#         self.patch_hw = (h, w)
#         self.num_patches = h * w

#     @torch.no_grad
#     def get_output_shape(self, img_size):
#         return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         # assert H == self.img_size[0] and W == self.img_size[1], \
#         #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x)
#         print("checking projector")
#         print(self.proj.weight.sum())
#         x = x.flatten(2).transpose(1, 2)
#         print("projector output")
#         print(x.sum())
#         return x


# """ PORTED FROM AUDIOMAE PREPROCESSING """

# NORM_STATS = {
#     "audioset": [-4.2677393, 4.5689974],
# }
# IS_MULTILABLE = {
#     "audioset": True,
# }
# AUDIO_CONF = {
#     "num_mel_bins": 128,
#     "freqm": 0,  # pretrain; finetune 48
#     "timem": 0,  # pretrain; finetune 192 TODO check how important is this difference
#     "ft_freqm": 48,  # pretrain; finetune 48
#     "ft_timem": 192,  # pretrain; finetune 192 TODO check how important is this difference
#     "mixup": 0,  # pretrain
#     "dataset": "audioset",
#     "mode": "train",
#     "mean": NORM_STATS["audioset"][0],
#     "std": NORM_STATS["audioset"][1],
#     "multilabel": IS_MULTILABLE["audioset"],
#     "noise": False,
# }


# class AudioPreprocessor:
#     """Handling the preprocessing of both 10s and 30s audios"""

#     def __init__(self, freqm, timem, target_length) -> None:
#         self.audio_conf = AUDIO_CONF
#         self.roll_mag_aug = False
#         self.freqm = freqm
#         self.timem = timem
#         self.target_length = target_length

#     def _roll_mag_aug(self):
#         raise NotImplementedError

#     def _wav2fbank(self, filename):
#         waveform, _, target_sr = load_resample_audio(filename, 16000)
#         waveform = waveform - waveform.mean()
#         if self.roll_mag_aug:
#             waveform = self._roll_mag_aug(waveform)
#         fbank = torchaudio.compliance.kaldi.fbank(
#             torch.tensor(waveform).unsqueeze(0),
#             htk_compat=True,
#             sample_frequency=target_sr,  # converting wrt target_sr
#             use_energy=False,
#             window_type="hanning",
#             num_mel_bins=self.audio_conf.get("num_mel_bins"),
#             dither=0.0,
#             frame_shift=10,
#         )
#         n_frames = fbank.shape[0]
#         p = self.target_length - n_frames
#         if p > 0:
#             m = torch.nn.ZeroPad2d((0, 0, 0, p))
#             fbank = m(fbank)
#         elif p < 0:
#             fbank = fbank[: self.target_length, :]
#         return fbank, 0

#     def preprocess(self, datum, *args, **kwargs):
#         fbank, _ = self._wav2fbank(datum["local_audio_path"])
#         freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
#         timem = torchaudio.transforms.TimeMasking(self.timem)
#         fbank = fbank.transpose(0, 1).unsqueeze(0)  # 1, 128, 1024 (..., freq, time)
#         if self.freqm != 0:
#             fbank = freqm(fbank)
#         if self.timem != 0:
#             fbank = timem(fbank)  # (..., freq, time)
#         fbank = torch.transpose(fbank.squeeze(), 0, 1)  # time, freq
#         fbank = (fbank - self.audio_conf["mean"]) / (self.audio_conf["std"] * 2)
#         if self.audio_conf["noise"]:  # default is false, true for spc
#             fbank = (
#                 fbank
#                 + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
#             )
#             fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
#         # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
#         return fbank.unsqueeze(0)


# class AudioMAEencoder(nn.Module):
#     """audio_tower_cfg: ModelArgs"""

#     def __init__(self, audio_tower_cfg, delay_load=False) -> None:
#         super().__init__()
#         if "vitb_pretrained.pth" in audio_tower_cfg.audio_pretrained_ckpt_path:
#             raise
#         elif "vitb_finetuned.pth" in audio_tower_cfg.audio_pretrained_ckpt_path:
#             # audio_mae.models_vit.VisionTransformer
#             mae = finetunedmae_vit_base_patch16(
#                 num_classes=527,  # default from main_finetune_as.py of audiomae
#                 drop_path_rate=0.1,
#                 global_pool=True,
#                 mask_2d=True,
#                 use_custom_patch=False,
#             )
#             # AudioMAE can only handle 512 patches due to position embedding
#             # so here /3 when defining and loading the model
#             # bagging behaviour adapt in forward
#             # audio_tower_cfg.audio_input_target_length 3072
#             mae.patch_embed = PatchEmbed_new_llava(
#                 img_size=(audio_tower_cfg.audio_input_target_length // 3, 128),
#                 patch_size=(16, 16),
#                 in_chans=1,
#                 embed_dim=768,
#                 stride=16,
#             )  # no overlap. stride=img_size=16
#             num_patches = mae.patch_embed.num_patches  # 512
#             mae.pos_embed = nn.Parameter(
#                 torch.zeros(1, num_patches + 1, 768), requires_grad=False
#             )  # fixed sin-cos embedding
#             self.mae = mae
#             freqm = AUDIO_CONF["ft_freqm"]
#             timem = AUDIO_CONF["ft_timem"]
#         else:
#             raise ValueError("Audio Tower type not supported.")

#         self.image_processor = AudioPreprocessor(
#             freqm=freqm,
#             timem=timem,
#             target_length=audio_tower_cfg.audio_input_target_length,  # 512*3
#         )
#         self.hidden_size = 768
#         self.is_loaded = False
#         if not delay_load:
#             print(
#                 f"\n=>Loading pretrained audio ckpt: {audio_tower_cfg.audio_pretrained_ckpt_path}"
#             )
#             ckpt = torch.load(
#                 audio_tower_cfg.audio_pretrained_ckpt_path, map_location="cpu"
#             )["model"]
#             msg = self.mae.load_state_dict(ckpt, strict=True)
#             self.is_loaded = True
#             print(f"\n=>Loaded pretrained audio ckpt. MSG: {msg}")
#         self.mae.requires_grad_(False)

#     def down_sample(self, x):
#         bs, hdim = x.shape[0], x.shape[-1]
#         x = x.view(bs, 64, -1, hdim).mean(dim=2)
#         return x

#     def timeframe_view(self, x, num_channels, num_bins):
#         x = x.view(-1, num_channels, 1024, num_bins)
#         return x

#     @torch.no_grad()
#     def forward(self, x):
#         # x: [16, 1, 3072, 128])
#         print("wocao!")
#         # print(x)
#         # tensor([[[[-0.0298, -0.3867, -0.0100,  ...,  0.2891,  0.3379,  0.1157],
#         #         [-0.2363, -0.6680, -0.2910,  ...,  0.3594,  0.2363,  0.1514],
#         #         [-0.0791, -0.3926, -0.0165,  ...,  0.3242,  0.2295,  0.1934],
#         #         ...,
#         #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668],
#         #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668],
#         #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668]]]],
#         bs, num_channels, num_bins = x.shape[0], x.shape[1], x.shape[-1]
#         x = self.timeframe_view(x, num_channels, num_bins)  # bsx3 x 1 x 1024 x 128
#         # print(x)
#         # tensor([[[[-0.0298, -0.3867, -0.0100,  ...,  0.2891,  0.3379,  0.1157],
#         #   [-0.2363, -0.6680, -0.2910,  ...,  0.3594,  0.2363,  0.1514],
#         #   [-0.0791, -0.3926, -0.0165,  ...,  0.3242,  0.2295,  0.1934],
#         #   ...,
#         #   [-0.4863, -1.0547, -0.6797,  ..., -0.1631, -0.1934, -0.1533],
#         #   [-0.5664, -0.7695, -0.3926,  ..., -0.1895, -0.0981, -0.1240],
#         #   [-0.5391, -0.7344, -0.3594,  ..., -0.1504, -0.1895, -0.2148]]],

#         # [[[-0.5781, -1.1484, -0.7695,  ..., -0.2402, -0.3047, -0.2832],
#         #   [-0.5156, -0.7500, -0.3750,  ..., -0.2637, -0.1865, -0.2852],
#         #   [-0.6992, -0.7227, -0.3457,  ..., -0.3301, -0.2393, -0.2656],
#         #   ...,
#         #   [-0.5547, -0.6914, -0.3125,  ...,  0.3066,  0.3555,  0.3301],
#         #   [-0.5977, -0.7812, -0.4062,  ...,  0.2852,  0.2168,  0.2334],
#         #   [-0.7266, -0.7383, -0.3594,  ...,  0.3008,  0.3125,  0.3027]]],

#         # [[[-0.4609, -0.6992, -0.3203,  ...,  0.2539,  0.2217,  0.2539],
#         #   [-0.5742, -0.7188, -0.3398,  ...,  0.2578,  0.3047,  0.3066],
#         #   [-0.5078, -0.6562, -0.2793,  ...,  0.2461,  0.1865,  0.1445],
#         #   ...,
#         #   [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668],
#         #   [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668],
#         #   [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668]]]],
#         x = self.mae.forward_features_no_pooling(x)  # bsx3, 513, 768
#         # print(x)
#         # tensor([[[ 0.2432, -0.2637,  0.0486,  ...,  0.1797,  0.1777, -0.1738],
#         #         [ 0.3965, -0.8398, -0.1807,  ...,  0.3652, -0.0623,  0.0247],
#         #         [ 1.0938, -0.1816, -0.0371,  ..., -0.1133, -0.4785,  0.2559],
#         #         ...,
#         #         [ 0.0564, -0.4824, -0.0552,  ...,  0.2578,  0.0723,  0.5586],
#         #         [ 0.2207, -0.2656, -0.8359,  ...,  0.5117, -0.0649,  0.0835],
#         #         [ 1.1172, -0.2598, -1.0469,  ...,  0.5234,  0.0962, -0.6562]],

#         #         [[ 0.2578, -0.2578,  0.0447,  ...,  0.1846,  0.1719, -0.1738],
#         #         [ 0.7695, -0.8906, -0.4707,  ...,  0.6250, -0.3027,  0.0122],
#         #         [ 0.5625, -0.9336, -0.2285,  ...,  0.1147,  0.8906, -0.1030],
#         #         ...,
#         #         [ 1.5156,  0.2031,  0.1108,  ...,  0.7539, -0.0938, -0.4141],
#         #         [ 1.1953,  0.8711,  0.2539,  ...,  0.7617,  0.0425, -0.0562],
#         #         [ 1.5312, -0.0025, -0.4277,  ...,  0.7578, -0.0067, -0.6055]],

#         #         [[ 0.2471, -0.2617,  0.0498,  ...,  0.1855,  0.1768, -0.1797],
#         #         [ 0.1177, -0.3184, -0.4766,  ...,  0.5234,  0.0031, -0.9180],
#         #         [ 1.1406,  0.1162,  0.2754,  ...,  0.3574, -0.3242, -0.5703],
#         #         ...,
#         #         [ 0.2090, -0.0830,  0.3066,  ...,  0.8203, -0.2871, -0.2910],
#         #         [ 0.3887, -0.0952,  0.3730,  ...,  0.8008, -0.1934, -0.4121],
#         #         [ 0.5781, -0.2314,  0.3809,  ...,  0.7930, -0.1846, -0.4180]]],
#         # print(x.shape)
#         # print("patch embedded")
#         # tensor([[[-0.1309, -0.0884, -0.0222,  ..., -0.2236, -0.1650,  0.1191],
#         #         [ 0.0693,  0.2188,  0.2969,  ...,  0.0251,  0.0293, -0.1504],
#         #         [ 0.1182, -0.1250,  0.3301,  ...,  0.0505, -0.1099, -0.1074],
#         #         ...,
#         #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
#         #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
#         #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400]],

#         #         [[ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
#         #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
#         #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
#         #         ...,
#         #         [-0.3223, -0.6211,  0.5273,  ...,  0.2197, -0.1562, -0.0776],
#         #         [-0.3691, -0.1035,  0.1113,  ...,  0.0107, -0.3652, -0.2773],
#         #         [-0.1738, -0.0176,  0.1484,  ..., -0.1572, -0.1279, -0.2695]],

#         #         [[-0.5938, -0.3203,  0.2246,  ..., -0.0608,  0.1250,  0.0410],
#         #         [-0.3418,  0.1055,  0.1426,  ...,  0.0981, -0.2090,  0.1924],
#         #         [-0.0527,  0.2852, -0.1221,  ...,  0.3281, -0.0664, -0.0269],
#         #         ...,
#         #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
#         #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
#         #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400]]],
#         # Parameter containing:
#         # tensor([[[[ 0.0786, -0.0908,  0.1060,  ...,  0.1270,  0.1367,  0.0864],
#         #           [-0.0388,  0.0053, -0.0693,  ...,  0.0312,  0.1055, -0.0530],
#         #           [ 0.1680, -0.0728,  0.0282,  ..., -0.0669, -0.0186, -0.0483],
#         #           ...,
#         #           [ 0.1377, -0.0520, -0.0070,  ...,  0.0703,  0.0204, -0.1235],
#         #           [ 0.0942,  0.1084,  0.1162,  ..., -0.0566,  0.0403, -0.0383],
#         #           [ 0.0713, -0.0986,  0.0014,  ..., -0.0288,  0.0204, -0.2500]]],

#         #         [[[-0.0977,  0.1128,  0.3066,  ...,  0.0068,  0.0640,  0.0297],
#         #           [-0.0239,  0.1143,  0.1416,  ...,  0.0981,  0.1011,  0.0889],
#         #           [-0.0417,  0.0398,  0.0015,  ...,  0.0593, -0.0108, -0.0415],
#         #           ...,
#         #           [ 0.0564, -0.0767,  0.0154,  ...,  0.0530,  0.0073,  0.1055],
#         #           [ 0.0908, -0.1416, -0.0284,  ..., -0.0176,  0.0510,  0.1084],
#         #           [ 0.1318, -0.0664, -0.0220,  ...,  0.0035, -0.1157,  0.0005]]],

#         #         [[[-0.0105, -0.0845, -0.0576,  ...,  0.0410,  0.0476, -0.0688],
#         #           [-0.0250, -0.0496,  0.0293,  ...,  0.0270, -0.0464, -0.0747],
#         #           [-0.0029, -0.0884,  0.0074,  ...,  0.0757,  0.0014, -0.0320],
#         #           ...,
#         #           [-0.1099,  0.1260,  0.0476,  ...,  0.0303, -0.0679, -0.0188],
#         #           [-0.0067,  0.1416, -0.0172,  ..., -0.0410, -0.0459, -0.1816],
#         #           [-0.1133,  0.1030,  0.0962,  ...,  0.1064,  0.0312,  0.0527]]],
#         exit()
#         x = x[:, 1:, :]
#         # x = self.down_sample(x[:, 1:, :])  # bs*3 x 64 x hidden
#         h_dim = x.shape[-1]
#         # print(x.view(bs, -1, 512, h_dim).shape)
#         x = x.view(bs, -1, 512, h_dim).mean(dim=1)
#         return x


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
        print("checking projector")
        print(self.proj.weight.sum())
        x = x.flatten(2).transpose(1, 2)
        print("projector output")
        print(x.sum())
        return x


# """ PORTED FROM AUDIOMAE PREPROCESSING """

NORM_STATS = {
    "audioset": [-4.2677393, 4.5689974],
}
TARGET_LENGTH = {"audioset": 1024}
IS_MULTILABLE = {
    "audioset": True,
}
AUDIO_CONF = {
    "num_mel_bins": 128,
    "target_length": TARGET_LENGTH["audioset"],
    "freqm": 0,  # pretrain; finetune 48
    "timem": 0,  # pretrain; finetune 192 TODO check how important is this difference
    "ft_freqm": 48,  # pretrain; finetune 48
    "ft_timem": 192,  # pretrain; finetune 192 TODO check how important is this difference
    "mixup": 0,  # pretrain
    "dataset": "audioset",
    "mode": "train",
    "mean": NORM_STATS["audioset"][0],
    "std": NORM_STATS["audioset"][1],
    "multilabel": IS_MULTILABLE["audioset"],
    "noise": False,
}


class AudioPreprocessor:
    def __init__(self, freqm, timem) -> None:
        self.audio_conf = AUDIO_CONF
        self.roll_mag_aug = False
        self.freqm = freqm
        self.timem = timem

    def _roll_mag_aug(self):
        raise NotImplementedError

    def _wav2fbank(self, filename):
        # waveform, sr = torchaudio.load(filename)  # NOTE: reduce the loading cost
        waveform, _, target_sr = load_resample_audio(filename, 16000)
        waveform = waveform - waveform.mean()
        if self.roll_mag_aug:
            waveform = self._roll_mag_aug(waveform)
        fbank = torchaudio.compliance.kaldi.fbank(
            torch.tensor(waveform).unsqueeze(0),
            htk_compat=True,
            sample_frequency=target_sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.audio_conf.get("num_mel_bins"),
            dither=0.0,
            frame_shift=10,
        )
        # TODO FIX THIS HACKING!
        # target_length = self.audio_conf.get("target_length")
        target_length = 1024 * 3
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
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
        if self.audio_conf["noise"]:  # default is false, true for spc
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank.unsqueeze(0)


class AudioMAEencoder(nn.Module):
    """audio_tower_cfg: ModelArgs"""

    def __init__(self, audio_tower_cfg, delay_load=False) -> None:
        super().__init__()
        if "vitb_pretrained.pth" in audio_tower_cfg.audio_pretrained_ckpt_path:
            self.mae = mae_vit_base_patch16_dec512d8b(
                in_chans=1,
                audio_exp=True,
                img_size=(1024, 128),
                decoder_mode=1,
                decoder_depth=16,
            )
            freqm = AUDIO_CONF["freqm"]
            timem = AUDIO_CONF["timem"]
        elif "vitb_finetuned.pth" in audio_tower_cfg.audio_pretrained_ckpt_path:
            mae = finetunedmae_vit_base_patch16(
                num_classes=527,  # default from main_finetune_as.py of audiomae
                drop_path_rate=0.1,
                global_pool=True,
                mask_2d=True,
                use_custom_patch=False,
            )
            mae.patch_embed = PatchEmbed_new_llava(
                img_size=(AUDIO_CONF["target_length"], 128),
                patch_size=(16, 16),
                in_chans=1,
                embed_dim=768,
                stride=16,
            )  # no overlap. stride=img_size=16
            num_patches = mae.patch_embed.num_patches
            mae.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, 768), requires_grad=False
            )  # fixed sin-cos embedding
            self.mae = mae
            freqm = AUDIO_CONF["ft_freqm"]
            timem = AUDIO_CONF["ft_timem"]
        else:
            raise ValueError("Audio Tower type not supported.")

        self.image_processor = AudioPreprocessor(freqm=freqm, timem=timem)
        self.hidden_size = 768
        self.is_loaded = False
        if not delay_load:
            print(
                f"\n=>Loading pretrained audio ckpt from {audio_tower_cfg.audio_pretrained_ckpt_path}"
            )
            ckpt = torch.load(
                audio_tower_cfg.audio_pretrained_ckpt_path, map_location="cpu"
            )["model"]
            if type(self.mae) != mae_vit_base_patch16_dec512d8b:
                pass
                # ckpt.pop("fc_norm.weight") # we have to use these
                # ckpt.pop("fc_norm.bias") # though this fc_norm is applied to bag features
            msg = self.mae.load_state_dict(ckpt, strict=True)
            self.is_loaded = True
            print(f"\n=>Loaded pretrained audio ckpt. MSG: {msg}")
        self.mae.requires_grad_(False)

    def down_sample(self, x):
        bs, hdim = x.shape[0], x.shape[-1]
        x = x.view(bs, 64, -1, hdim).mean(dim=2)
        return x

    def timeframe_view(self, x, bs, num_channels, num_bins):
        # TODO this should be compatible with both 10s and 30s, in a hacking way
        x = x.view(-1, num_channels, 1024, num_bins)
        return x

    @torch.no_grad()
    def forward(self, x):
        if type(self.mae) == mae_vit_base_patch16_dec512d8b:
            # x: [16, 1, 3072, 128])
            bs, num_channels, num_bins = x.shape[0], x.shape[1], x.shape[-1]
            x = self.timeframe_view(x, bs, num_channels, num_bins)
            all_hidden = self.mae.forward_encoder_no_mask(x)
            all_hidden = self.down_sample(all_hidden[:, 1:, :])
            h_dim = all_hidden.shape[-1]
            all_hidden = all_hidden.view(bs, -1, 64, h_dim).mean(dim=1)
            # return all_hidden[:, 1:, :]  # not using the CLS token, but all tokens
            # print(all_hidden.shape) # bs x 513 x 768
            return all_hidden
        else:
            print("hhhhhhhhhhhhhhhha")
            # x: [16, 1, 3072, 128])
            # print(x)
            # tensor([[[[-0.0298, -0.3867, -0.0100,  ...,  0.2891,  0.3379,  0.1157],
            #         [-0.2363, -0.6680, -0.2910,  ...,  0.3594,  0.2363,  0.1514],
            #         [-0.0791, -0.3926, -0.0165,  ...,  0.3242,  0.2295,  0.1934],
            #         ...,
            #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668],
            #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668],
            #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668]]]],
            #     device='cuda:0', dtype=torch.bfloat16)
            bs, num_channels, num_bins = x.shape[0], x.shape[1], x.shape[-1]
            x = self.timeframe_view(x, bs, num_channels, num_bins)
            # print(x)
            # tensor([[[[-0.0298, -0.3867, -0.0100,  ...,  0.2891,  0.3379,  0.1157],
            #         [-0.2363, -0.6680, -0.2910,  ...,  0.3594,  0.2363,  0.1514],
            #         [-0.0791, -0.3926, -0.0165,  ...,  0.3242,  0.2295,  0.1934],
            #         ...,
            #         [-0.4863, -1.0547, -0.6797,  ..., -0.1631, -0.1934, -0.1533],
            #         [-0.5664, -0.7695, -0.3926,  ..., -0.1895, -0.0981, -0.1240],
            #         [-0.5391, -0.7344, -0.3594,  ..., -0.1504, -0.1895, -0.2148]]],

            #         [[[-0.5781, -1.1484, -0.7695,  ..., -0.2402, -0.3047, -0.2832],
            #         [-0.5156, -0.7500, -0.3750,  ..., -0.2637, -0.1865, -0.2852],
            #         [-0.6992, -0.7227, -0.3457,  ..., -0.3301, -0.2393, -0.2656],
            #         ...,
            #         [-0.5547, -0.6914, -0.3125,  ...,  0.3066,  0.3555,  0.3301],
            #         [-0.5977, -0.7812, -0.4062,  ...,  0.2852,  0.2168,  0.2334],
            #         [-0.7266, -0.7383, -0.3594,  ...,  0.3008,  0.3125,  0.3027]]],

            #         [[[-0.4609, -0.6992, -0.3203,  ...,  0.2539,  0.2217,  0.2539],
            #         [-0.5742, -0.7188, -0.3398,  ...,  0.2578,  0.3047,  0.3066],
            #         [-0.5078, -0.6562, -0.2793,  ...,  0.2461,  0.1865,  0.1445],
            #         ...,
            #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668],
            #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668],
            #         [ 0.4668,  0.4668,  0.4668,  ...,  0.4668,  0.4668,  0.4668]]]],
            all_hidden = self.mae.forward_features_no_pooling(x)
            # print(all_hidden)
            #         tensor([[[ 0.2598, -0.2617,  0.0508,  ...,  0.1846,  0.1738, -0.1699],
            #     [ 0.4453, -0.8750, -0.1182,  ...,  0.3555, -0.1592, -0.0074],
            #     [ 1.0234, -0.2852, -0.0977,  ..., -0.1348, -0.5391,  0.2539],
            #     ...,
            #     [ 0.5117, -0.6211, -0.1387,  ...,  0.3262, -0.3125, -0.6328],
            #     [ 0.5859, -0.0781, -0.8242,  ...,  0.7266,  0.2754,  0.0723],
            #     [ 1.1484, -0.1494, -1.1719,  ...,  0.7070,  0.2334, -0.4961]],

            #     [[ 0.2617, -0.2617,  0.0493,  ...,  0.1855,  0.1699, -0.1689],
            #     [ 0.6836, -0.8633, -0.4180,  ...,  0.6211, -0.3066, -0.0139],
            #     [ 0.4434, -0.8516, -0.0474,  ...,  0.0728,  0.8047, -0.0688],
            #     ...,
            #     [ 0.5820, -0.0879,  0.4043,  ...,  0.6875, -0.3457, -0.9219],
            #     [ 1.2344,  0.1904,  0.1953,  ...,  0.5430,  0.0942, -0.5352],
            #     [ 1.7344,  0.1260, -0.2969,  ...,  0.7695, -0.1396, -0.5469]],

            #     [[ 0.2520, -0.2617,  0.0493,  ...,  0.1836,  0.1738, -0.1729],
            #     [ 0.1387, -0.2500, -0.5039,  ...,  0.4512, -0.0203, -0.9453],
            #     [ 1.0469,  0.1914,  0.2930,  ...,  0.3105, -0.2754, -0.4727],
            #     ...,
            #     [ 0.3496,  0.2383,  0.1914,  ...,  0.8945, -0.1455, -0.1187],
            #     [ 0.4355,  0.0071,  0.3105,  ...,  0.9648, -0.2930, -0.2930],
            #     [ 0.6172, -0.2100,  0.4688,  ...,  0.9258, -0.1191, -0.3223]]],
            # device='cuda:0', dtype=torch.bfloat16)
            # print(x.shape)
            # print("patch embedded")
            # tensor([[[-0.4023, -0.2227, -0.0762,  ..., -0.4316, -0.0381, -0.0928],
            #         [ 0.2656,  0.1758,  0.1816,  ...,  0.0405,  0.0176, -0.2168],
            #         [ 0.1182, -0.1250,  0.3301,  ...,  0.0505, -0.1099, -0.1074],
            #         ...,
            #         [ 0.0586, -0.2715,  0.1699,  ...,  0.1396, -0.1914, -0.0092],
            #         [-0.2773, -0.2910,  0.2695,  ..., -0.0728, -0.1289, -0.1084],
            #         [-0.4785,  0.0713,  0.5195,  ..., -0.1680,  0.2207, -0.0474]],

            #         [[-0.8203, -0.6211,  0.2500,  ..., -0.4023,  0.2383,  0.0879],
            #         [ 0.0625, -0.6797,  0.0066,  ..., -0.3789, -0.1191, -0.8516],
            #         [-0.7266,  0.7227,  0.4551,  ..., -0.1055, -0.0698,  0.0352],
            #         ...,
            #         [-0.3223, -0.6211,  0.5273,  ...,  0.2197, -0.1562, -0.0776],
            #         [-0.3691, -0.1035,  0.1113,  ...,  0.0107, -0.3652, -0.2773],
            #         [-0.1738, -0.0176,  0.1484,  ..., -0.1572, -0.1279, -0.2695]],

            #         [[-0.8359, -0.3750,  0.2539,  ..., -0.3281,  0.2734,  0.1182],
            #         [-0.0425, -0.0591, -0.0044,  ...,  0.0583, -0.3496,  0.0308],
            #         [-0.0527,  0.2852, -0.1221,  ...,  0.3281, -0.0664, -0.0269],
            #         ...,
            #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
            #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400],
            #         [ 0.1084, -0.0552,  0.1221,  ...,  0.0322, -0.1377,  0.0400]]],
            #     device='cuda:0', dtype=torch.bfloat16)
            # Parameter containing:
            # tensor([[[[ 0.0786, -0.0908,  0.1060,  ...,  0.1270,  0.1367,  0.0864],
            #         [-0.0388,  0.0053, -0.0693,  ...,  0.0312,  0.1055, -0.0530],
            #         [ 0.1680, -0.0728,  0.0282,  ..., -0.0669, -0.0186, -0.0483],
            #         ...,
            #         [ 0.1377, -0.0520, -0.0070,  ...,  0.0703,  0.0204, -0.1235],
            #         [ 0.0942,  0.1084,  0.1162,  ..., -0.0566,  0.0403, -0.0383],
            #         [ 0.0713, -0.0986,  0.0014,  ..., -0.0288,  0.0204, -0.2500]]],

            #         [[[-0.0977,  0.1128,  0.3066,  ...,  0.0068,  0.0640,  0.0297],
            #         [-0.0239,  0.1143,  0.1416,  ...,  0.0981,  0.1011,  0.0889],
            #         [-0.0417,  0.0398,  0.0015,  ...,  0.0593, -0.0108, -0.0415],
            #         ...,
            #         [ 0.0564, -0.0767,  0.0154,  ...,  0.0530,  0.0073,  0.1055],
            #         [ 0.0908, -0.1416, -0.0284,  ..., -0.0176,  0.0510,  0.1084],
            #         [ 0.1318, -0.0664, -0.0220,  ...,  0.0035, -0.1157,  0.0005]]],

            #         [[[-0.0105, -0.0845, -0.0576,  ...,  0.0410,  0.0476, -0.0688],
            #         [-0.0250, -0.0496,  0.0293,  ...,  0.0270, -0.0464, -0.0747],
            #         [-0.0029, -0.0884,  0.0074,  ...,  0.0757,  0.0014, -0.0320],
            #         ...,
            #         [-0.1099,  0.1260,  0.0476,  ...,  0.0303, -0.0679, -0.0188],
            #         [-0.0067,  0.1416, -0.0172,  ..., -0.0410, -0.0459, -0.1816],
            #         [-0.1133,  0.1030,  0.0962,  ...,  0.1064,  0.0312,  0.0527]]],
            # exit()
            all_hidden = self.down_sample(all_hidden[:, 1:, :])  # bs*3 x 64 x hidden
            h_dim = all_hidden.shape[-1]
            all_hidden = all_hidden.view(bs, -1, 64, h_dim).mean(dim=1)
            return all_hidden
