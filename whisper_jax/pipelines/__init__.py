from .long_audio import FlaxWhisperPmapPipeline as FlaxWhisperLongAudioPipeline
from .long_audio import create_pipeline as create_long_audio_pipeline
from .long_audio_seek import create_pipeline as create_long_audio_seek_pipeline
from .online import FlaxWhisperOnlinePipeline
from .short_audio import FlaxWhisperPjitPipeline, FlaxWhisperPmapPipeline, create_pipeline

FlaxWhisperPipeline = FlaxWhisperPmapPipeline
FlaxWhisperPipline = FlaxWhisperPmapPipeline

__all__ = [
    "FlaxWhisperPmapPipeline",
    "FlaxWhisperPjitPipeline",
    "FlaxWhisperPipeline",
    "FlaxWhisperPipline",
    "FlaxWhisperLongAudioPipeline",
    "FlaxWhisperOnlinePipeline",
    "create_pipeline",
    "create_long_audio_pipeline",
    "create_long_audio_seek_pipeline",
]
