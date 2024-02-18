import dataclasses
import os
import re
import signal
import subprocess
import time
import uuid
import shutil
import json
from datetime import datetime
from pprint import pformat

import torch
import requests
from urllib.parse import urlparse

import psutil
import pathlib
import logging
import hashlib

from configs.default import get_cfg_defaults
from talking_head.params import Task, LaunchOptions, TaskParams
from talking_head.dirs import get_task_dir
from sadtalker import inference

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def download(resource_url, target_dir, default_ext):
    if not resource_url.startswith('http'):
        raise Exception(f'must be url: {resource_url}')
    resource_path = urlparse(resource_url).path
    resource_name = os.path.basename(resource_path)
    if '.' not in resource_name and default_ext is not None:
        resource_name = f'{resource_name}.{default_ext}'
    full_path = f'{target_dir}/{resource_name}'
    with requests.get(resource_url, stream=True) as res:
        with open(full_path, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    return full_path

def build_params(args, config='./arg_config.json'):
    with open(config, 'r') as file:
        config_dict = json.load(file)
    for key, value in config_dict.items():
        if key=='preprocess':
            print(key,args.preprocess)
        if not hasattr(args, key):
            if not isinstance(value, dict):
                setattr(args, key, value)
            elif 'default' in value:
                setattr(args, key, value['default'])
        elif isinstance(value, dict) and 'need_change_url' in value:
            local_url = download(getattr(args,key), args.task_dir, value['default_ext'])
            setattr(args, key, local_url)
            if 'sys_name' in value:
                setattr(args, value['sys_name'], local_url)
        elif isinstance(value, bool):
                str_val = getattr(args, key)
                if str_val == '1' or str_val.lower() == 'ture':
                    setattr(args, key, True)
                else:
                    setattr(args, key, False)
    return args

def launch(config, task: Task, launch_options: LaunchOptions, logger=None):
    if logger is None:
        logger = logging.getLogger('launch')

    prepare_start_at = datetime.now().isoformat()

    # logger.info(pformat(task))
    # logger.info(pformat(launch_options))
    params = TaskParams(**dataclasses.asdict(task), **dataclasses.asdict(launch_options))
    if launch_options.device_index is not None:
        params.device = f'cuda:{launch_options.device_index}'
    else:
        logger.warning('device_index not set')
        params.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TASKS_DIR = config['TASKS_DIR']
    task_dir = get_task_dir(TASKS_DIR, task.task_id, task.subdir)
    os.makedirs(task_dir, exist_ok=True)
    params.task_dir = task_dir

    # params.img_crop = True
    # params.image_path = download(task.image_url, task_dir, 'jpg')
    # params.audio_path = download(task.audio_url, task_dir, 'm4a')

    # style_name = params.style_name
    # if style_name is None:
    #     style_name = 'M030_front_neutral_level1_001'
    #     # style_name = 'W009_front_neutral_level1_001'
    #     # style_name = 'W011_front_neutral_level1_001'
    # params.style_clip_path = f'data/style_clip/3DMM/{style_name}.mat'
    # params.pose_path = 'data/pose/RichardShelby_front_neutral_level1_001.mat'
    # if params.max_gen_len is None:
    #     params.max_gen_len = 600
    # if params.cfg_scale is None:
    #     params.cfg_scale = 2.0

    # model_cfg = get_cfg_defaults()
    # model_cfg.CF_GUIDANCE.SCALE = params.cfg_scale
    # model_cfg.freeze()

    json.dump(dataclasses.asdict(params), open(f'{task_dir}/params.json', 'w'), indent=2)

    result_file = os.path.join(task_dir, 'result.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    result = {
        'prepare_start_at': prepare_start_at,
        'inference_start_at': datetime.now().isoformat()
    }
    try:
        inference(params)
        result['success'] = True
        result['cropped_image_path'] = params.cropped_image_path
        result['output_video_path'] = params.output_video_path
        # upload ...
    except Exception as e:
        logger.error(e)
        result['success'] = False
        result['error_message'] = str(e)
    result['finish_at'] = datetime.now().isoformat()

    json.dump(result, open(result_file, 'w'), indent=2)

    logger.info(pformat(result))
    logger.info(f'Finished')

    return {'success': True}
