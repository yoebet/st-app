import dataclasses
import os
import random
import sys
import ipaddress
import traceback
import subprocess
from threading import Thread
import multiprocessing
from contextlib import redirect_stdout, redirect_stderr
import shutil
import json
import time
from datetime import datetime
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter
import requests
from urllib.parse import urlparse
import logging
from PIL import Image
import numpy as np

from sadtalker import inference
from talking_head.params import Params
from talking_head.dirs import get_task_dir, get_tf_logging_dir

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


def verify_url(url):
    import socket
    from urllib.parse import urlparse
    try:
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        host = socket.gethostbyname_ex(domain_name)
        for ip in host[2]:
            ip_addr = ipaddress.ip_address(ip)
            if not ip_addr.is_global:
                return False
    except Exception:
        return False

    return True


def download(resource_url, target_dir, filename, default_ext):
    if not resource_url.startswith('http'):
        raise Exception(f'must be url: {resource_url}')
    # if not verify_url(resource_url):
    #     raise Exception(f'local resource not allowed')
    resource_path = urlparse(resource_url).path
    resource_name = os.path.basename(resource_path)
    base_name, ext = os.path.splitext(resource_name)
    if filename is None:
        filename = base_name
    if ext is None:
        ext = default_ext
    elif ext == '.jfif':
        ext = '.jpg'
    if ext is not None:
        filename = f'{filename}{ext}'

    full_path = f'{target_dir}/{filename}'
    with requests.get(resource_url, stream=True) as res:
        with open(full_path, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    return full_path


def tf_log_img(writer: SummaryWriter, tag, image_path, global_step=0):
    img = Image.open(image_path)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    np_image = np.asarray(img)
    writer.add_image(tag, np_image, global_step, dataformats="HWC")


def run_sync( params: Params,*, logger, result_file: str, result, log_file: str):
    if result is None:
        result = {}

    with open(log_file, 'w') as lf:
        with redirect_stdout(lf), redirect_stderr(lf):
            try:
                result['inference_start_at'] = datetime.now().isoformat()

                inference(params)

                result['success'] = True
                result['cropped_image_file'] = os.path.basename(params.cropped_image_path)
                result['output_video_file'] = os.path.basename(params.output_video_path)
                # result['output_video_duration'] = params.output_video_duration

            except Exception as e:
                print(str(e), file=sys.stderr)
                traceback.print_exc()
                result['success'] = False
                result['error_message'] = str(e)
            result['finished_at'] = datetime.now().isoformat()

            json.dump(result, open(result_file, 'w'), indent=2)

    if result['success']:
        logger.info(f'task {params.task_id} Finished.')
    else:
        logger.error(f'task {params.task_id} Failed: {result["error_message"]}')

    torch.cuda.empty_cache()
    return result

def build_params(args, config='./arg_config.json'):
    with open(config, 'r') as file:
        config_dict = json.load(file)
    for key, value in config_dict.items():
        if not hasattr(args, key):
            if not isinstance(value, dict):
                setattr(args, key, value)
            elif 'default' in value:
                setattr(args, key, value['default'])
        elif isinstance(value, dict) and 'need_change_url' in value:
            local_url = download(getattr(args,key), args.task_dir, value['default_file_name'] if 'default_file_name' in value else None,value['default_ext'])
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
def launch(config, task: Params, launch_options: Params, logger=None):
    if logger is None:
        logger = logging.getLogger('launch')

    prepare_start_at = datetime.now().isoformat()

    # logger.info(pformat(task))
    # logger.info(pformat(launch_options))
    params = task.merge(launch_options)
    if torch.cuda.is_available():
        if hasattr(launch_options, 'device_index'):
            params.device = f'cuda:{launch_options.device_index}'

            free, total = torch.cuda.mem_get_info(launch_options.device_index)
            g = 1024 ** 3
            if free < 15 * g:
                logger.warning(f'{params.task_id}: device occupied')
                return {
                    'success': False,
                    'error_message': 'device occupied',
                }
        else:
            logger.warning('device_index not set')
            params.device = 'cuda'
    else:
        params.device = 'cpu'

    TASKS_DIR = config['TASKS_DIR']
    task_dir = get_task_dir(TASKS_DIR, task.task_id, task.sub_dir)
    os.makedirs(task_dir, exist_ok=True)
    params.task_dir = task_dir
    params = build_params(params)
    json.dump(vars(params), open(f'{task_dir}/params.json', 'w'), indent=2)

    result_file = os.path.join(task_dir, 'result.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    result = {
        'prepare_start_at': prepare_start_at,
    }
    log_file = os.path.join(task_dir, f'log-{str(int(time.time()))}.txt')
    logger.info(f'Logging to {log_file} ...')

    res = {'success': True, }
    args = (params)
    kwargs = {'result': result, 'result_file': result_file, 'log_file': log_file, 'logger': logger}

    try:
        if params.run_mode == 'sync':
            res = run_sync(params, **kwargs)
        elif params.run_mode == 'process':
            process = multiprocessing.Process(target=run_sync, args=args, kwargs=kwargs)
            process.start()
            res['pid'] = process.pid
        else:  # thread
            thread_name = f'thread_{params.task_id}_{random.randint(1000, 9990)}'
            # res['thread_name'] = thread_name
            thread = Thread(target=run_sync, args=args, kwargs=kwargs, name=thread_name)
            thread.start()
    except Exception as e:
        logger.error(e)
        res['success'] = False
        res['error_message'] = str(e)

    return res
