# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models" 
# available at https://arxiv.org/abs/2301.10226
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

from argparse import Namespace
args = Namespace()

import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(torch.cuda.is_available())           # Should print True if CUDA is available
print(torch.version.cuda)                  # Should print the CUDA version compatible with PyTorch
print(torch.cuda.get_device_name(0))       # Should display "GeForce GTX 1050"

arg_dict = {
    'run_gradio': True, 
    'demo_public': False, 
    'model_name_or_path': 'gpt2-medium',
    # 'model_name_or_path': 'facebook/opt-350m', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    # 'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': 700, 
    'max_new_tokens': 600, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.8, 
    'delta': 2.0, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 4.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'number_clusters': 200,
    'seed_separately': True,
}

args.__dict__.update(arg_dict)

from demo_watermark import main

main(args)
