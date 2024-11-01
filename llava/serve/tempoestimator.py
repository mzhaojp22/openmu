from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor


def EstimateTempo():
    wav = load_audio(AUDIO_FILE)
    beat_proc, tempo_proc = RNNBeatProcessor(), TempoEstimationProcessor(fps=100)
    beat_acts, tempo_acts = beat_proc(wav), tempo_proc(beat_acts)
    tempo_est = round(tempo_acts[0][0], 1)
    return tempo_est
