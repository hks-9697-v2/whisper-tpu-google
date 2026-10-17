# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

__version__ = "0.0.1"

from . import distributed, inputs, kernels, layers, models, pipelines
from .distributed import InferenceState, PjitPartitioner
from .models import FlaxWhisperForConditionalGeneration
from .pipelines import (
    FlaxWhisperLongAudioPipeline,
    FlaxWhisperOnlinePipeline,
    FlaxWhisperPjitPipeline,
    FlaxWhisperPmapPipeline,
    create_long_audio_pipeline,
    create_long_audio_seek_pipeline,
    create_pipeline,
)
from .pipelines import (
    long_audio as _long_audio_mod,
    long_audio_seek as _long_audio_seek_mod,
    online as _online_mod,
    short_audio as _short_audio_mod,
)
from .models import whisper as _whisper_mod
from .distributed import partitioner as _partitioner_mod, train_state as _train_state_mod
from .inputs import audio_loader as _audio_loader_mod
from .kernels.splash_attention import adapter as _splash_adapter_mod
from .kernels.flywheel import adapter as _flywheel_adapter_mod
from .kernels.ragged_paged_attention import (
    adapter as _rpa_adapter_mod,
    kernel_hd64 as _rpa_kernel_mod,
)

# Backward-compatible module aliases for legacy import paths
sys.modules[__name__ + ".pipeline"] = _short_audio_mod
sys.modules[__name__ + ".pipeline_long_audio"] = _long_audio_mod
sys.modules[__name__ + ".pipeline_long_audio_seek"] = _long_audio_seek_mod
sys.modules[__name__ + ".pipeline_online"] = _online_mod
sys.modules[__name__ + ".modeling_flax_whisper"] = _whisper_mod
sys.modules[__name__ + ".partitioner"] = _partitioner_mod
sys.modules[__name__ + ".train_state"] = _train_state_mod
sys.modules[__name__ + ".rust_audio_loader"] = _audio_loader_mod
sys.modules[__name__ + ".splash_adapter"] = _splash_adapter_mod
sys.modules[__name__ + ".flywheel_adapter"] = _flywheel_adapter_mod
sys.modules[__name__ + ".ragged_paged_adapter"] = _rpa_adapter_mod
sys.modules[__name__ + ".tokamax_rpa_hd64"] = _rpa_kernel_mod
