from typing import Any, Dict
import librosa
from tqdm import tqdm
from madmom.features.beats import RNNBeatProcessor
from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor,
)
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.tempo import TempoEstimationProcessor
from madmom.processors import SequentialProcessor


class ExtractMadmomKeyEstimates(object):
    def __init__(self):
        self.key_proc = CNNKeyRecognitionProcessor()

    def process(self, elem: Dict[str, Any]):
        key_acts = self.key_proc(elem["audio"])
        key_est = key_prediction_to_label(key_acts)
        elem["key"] = key_est
        return [elem]


class ExtractMadmomChordEstimates(object):
    def __init__(self, fps: int = 10):
        self.fps = fps
        featproc = CNNChordFeatureProcessor()
        decode = CRFChordRecognitionProcessor(fps=self.fps)
        self.chordrec = SequentialProcessor([featproc, decode])

    def process(self, elem: Dict[str, Any]):
        chord_est = self.chordrec(elem["audio"])
        # Postprocessing: quantize estimates; they are already fixed to a 0.1-sec grid when fps=10
        # but numpy stores in higher precision which makes the strings harder to read.
        # Also parse the major/minor chord shorthand.
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
        elem["chords"] = chord_est
        return [elem]


class ExtractMadmomDownbeatFeatures(object):
    def __init__(self, fps=100, beats_per_bar=[3, 4]):
        self.fps = fps
        self.beats_per_bar = beats_per_bar
        downbeat_decode = DBNDownBeatTrackingProcessor(
            beats_per_bar=self.beats_per_bar, fps=self.fps
        )
        downbeat_process = RNNDownBeatProcessor()
        self.downbeat_rec = SequentialProcessor([downbeat_process, downbeat_decode])

    def process(self, elem: Dict[str, Any]):
        downbeats_est = self.downbeat_rec(elem["audio"])

        # make the outputs more human-readable for GPT
        downbeats_est = [
            {"time": x[0], "beat_number": int(x[1])} for x in downbeats_est.tolist()
        ]

        elem["downbeats_madmom"] = downbeats_est
        return [elem]


class ExtractMadmomTempoFeatures(object):
    def __init__(self, fps=100):
        self.fps = fps
        self.beat_proc = RNNBeatProcessor()
        self.tempo_proc = TempoEstimationProcessor(fps=self.fps)

    def process(self, elem: Dict[str, Any]):
        beat_acts = self.beat_proc(elem["audio"])
        tempo_acts = self.tempo_proc(beat_acts)
        tempo_est = round(tempo_acts[0][0], 1)
        elem["tempo_in_beats_per_minute_madmom"] = tempo_est
        return [elem]
