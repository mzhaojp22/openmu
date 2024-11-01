import argparse
import torch
import transformers
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    process_audio,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

##########
# Added for tool invoking
from madmom.features.beats import RNNBeatProcessor
from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor,
)
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.tempo import TempoEstimationProcessor
from madmom.processors import SequentialProcessor
import torchaudio
import numpy as np
from scipy.signal import resample_poly
from .invoking_tools import invoke_tools


def load_resample_audio(file_path, target_sr):
    x, sr = torchaudio.load(file_path, normalize=True, channels_first=True)
    if x.ndim > 1:
        x = x.mean(dim=0, keepdim=False)
    x = x.numpy()
    if sr != target_sr:
        x = np.float32(resample_poly(x, target_sr, sr))
    return np.float32(x), sr, target_sr


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_audio(audio_file):
    return {"local_audio_path": audio_file}


def main(args):
    AUDIO_FILE = args.image_file

    # def GetAudioTempo():
    def F2():
        audiofile, _, _ = load_resample_audio(AUDIO_FILE, 44100)
        fps = 100
        beat_proc = RNNBeatProcessor()
        tempo_proc = TempoEstimationProcessor(fps=fps)
        beat_acts = beat_proc(audiofile)
        tempo_acts = tempo_proc(beat_acts)
        tempo_est = round(tempo_acts[0][0], 1)
        return tempo_est

    # def GetAudioKey():
    def F3():
        audiofile, _, _ = load_resample_audio(AUDIO_FILE, 44100)
        key_proc = CNNKeyRecognitionProcessor()
        key_acts = key_proc(audiofile)
        key_est = key_prediction_to_label(key_acts)
        return key_est

    # def GetAudioDownbeat():
    def F4():
        audiofile, _, _ = load_resample_audio(AUDIO_FILE, 44100)
        fps = 100
        beats_per_bar = [3, 4]
        downbeat_decode = DBNDownBeatTrackingProcessor(
            beats_per_bar=beats_per_bar, fps=fps
        )
        downbeat_process = RNNDownBeatProcessor()
        downbeat_rec = SequentialProcessor([downbeat_process, downbeat_decode])
        downbeats_est = downbeat_rec(audiofile)
        downbeats_est = [
            {"time": x[0], "beat_number": int(x[1])} for x in downbeats_est.tolist()
        ]
        return downbeats_est

    # def GetAudioChords():
    def F1(*args, **kwargs):
        audiofile, _, _ = load_resample_audio(AUDIO_FILE, 44100)
        fps = 10
        featproc = CNNChordFeatureProcessor()
        decode = CRFChordRecognitionProcessor(fps=fps)
        chordrec = SequentialProcessor([featproc, decode])
        chord_est = chordrec(audiofile)
        chord_est = [
            {
                "start_time": round(x[0], 1),
                "end_time": round(x[1], 1),
                "chord": (
                    x[2].replace(":maj", "major").replace(":min", "minor")
                    if x[2] != "N"
                    else "no chord"
                ),
            }
            for x in chord_est.tolist()
        ]
        return chord_est

    # function_registry = dict(
    #     GetAudioChords=GetAudioChords,
    #     GetAudioTempo=GetAudioTempo,
    #     GetAudioKey=GetAudioKey,
    #     GetAudioDownbeat=GetAudioDownbeat,
    # )

    function_registry = dict(
        F1=F1,
        F2=F2,
        F3=F3,
        F4=F4,
    )

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # llama3-stage-2-trail-llava-lora-epoch10

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        is_audio_model=args.is_audio_model,
        audio_ckpt=args.audio_ckpt,
        device=args.device,
    )
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "llama-3" in model_name.lower() or "llama3" in model_name.lower():
        conv_mode = "llama_3"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles
    image = load_audio(args.image_file)  # {"local_audio_path": audio_file}
    image_size = (1024 * 3, 128)
    image_tensor = process_audio([image], image_processor, model.config)  # 1x3072x128
    image_tensor = image_tensor.unsqueeze(0)

    if type(image_tensor) is list:
        image_tensor = [
            image.to(model.device, dtype=torch.float32) for image in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float32)

    model = model.to(torch.bfloat16).cuda()

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(model.device)
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.cuda(),
                images=image_tensor.to(torch.bfloat16).to(model.device),
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # print("before", outputs)
        outputs = invoke_tools(function_registry, outputs)
        print("after", outputs)
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        # refresh the conversage as it is not a dialog model
        conv = conv_templates[args.conv_mode].copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--is-audio-model", action="store_true", default=False)
    parser.add_argument(
        "--audio-ckpt",
        type=str,
        default="vitb_finetuned.pth",
    )
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
