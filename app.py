import os
import re
from pprint import pformat
import base64
import shutil
import json
import torch
from flask import Flask, jsonify, request, Response, abort, send_file, send_from_directory
from dotenv import dotenv_values
from talking_head.params import Params
from talking_head.dirs import get_task_dir
from launch import launch

app = Flask(__name__)

app.config.from_mapping(dotenv_values())
app.config.from_mapping(dotenv_values('.env.local'))

logger = app.logger


@app.route('/', methods=('GET',))
def index():
    return 'ok'


@app.before_request
def before_request_callback():
    path = request.path
    if path != '/':
        auth = request.headers.get('AUTHORIZATION')
        if not auth == app.config['AUTHORIZATION']:
            pass
            # abort(400)

@app.route('/<path:filename>')
def serve_file(filename):
    root_dir = './examples'
    return send_from_directory(root_dir, filename)

def trans_unit(bytes, unit):
    if unit is None:
        return bytes
    k = 1024
    m = k * k
    div = {'B': 1, 'K': k, 'M': m, 'G': k * m}.get(unit.upper())
    return bytes / div


@app.route('/check_mem_all/available', methods=('GET',))
def check_mem_all():
    unit = request.args.get('unit')
    import accelerate
    d = accelerate.utils.get_max_memory()
    pairs = [(i, trans_unit(n, unit)) for i, n in d.items()]
    return jsonify(pairs)


@app.route('/check_mem/<device_index>', methods=('GET',))
def check_device_mem(device_index):
    device_index = int(device_index)
    unit = request.args.get('unit')
    free, total = torch.cuda.mem_get_info(device_index)
    return jsonify({
        'free': trans_unit(free, unit),
        'total': trans_unit(total, unit),
    })


@app.route('/launch', methods=('POST',))
def launch_task():
    req = request.get_json()
    logger.info(pformat(req))
    task_params = req.get('task')
    launch_params = req.get('launch')
    if launch_params is None:
        launch_params = {}
    task = Params(task_params)
    launch_options = Params(launch_params)

    if hasattr(launch_options, 'device_index'):
        free, total = torch.cuda.mem_get_info(launch_options.device_index)
        k = 1024
        if free < 10 * k * k * k:
            return (jsonify({
                'success': False,
                'error_message': 'device occupied',
            }))

    try:
        launch_result = launch(app.config, task, launch_options, logger=logger)
    except Exception as e:
        logger.error(e)
        return jsonify({
            'success': False,
            'error_message': f"[launch] {type(e)}: {e}"
        })

    return jsonify(launch_result)


@app.route('/task/<task_id>/status', methods=('POST',))
def check_task_status(task_id):
    req = request.get_json()
    pid = req.get('root_pid')
    sub_dir = req.get('sub_dir')

    import psutil

    running = False

    if pid is not None:
        pid = int(pid)
    if pid is not None and psutil.pid_exists(pid):
        rp = psutil.Process(pid)
        pname = rp.name()
        logger.info(pname)
        if 'accelerate' not in pname and 'python' not in pname:
            # raise Exception(f'wrong pid: {pid}, {pname}')
            return jsonify({
                'success': True,
                'task_status': 'failed',
                'failure_reason': 'wpn'
            })
        try:
            rp.cmdline()
        except psutil.ZombieProcess:
            pass
        except psutil.AccessDenied:
            logger.error('AccessDenied')
            return jsonify({
                'success': False,
                'error_message': 'wrong pid'
            })

        pstatus = rp.status()
        if pstatus == 'running':
            running = True
        elif pstatus == 'zombie':
            pass
        else:
            logger.info(f'process status: {pstatus}')
            running = True

    if running:
        return jsonify({
            'success': True,
            'task_status': 'running',
        })

    TASKS_DIR = app.config['TASKS_DIR']
    task_dir = get_task_dir(TASKS_DIR, task_id, sub_dir)

    if os.path.isdir(task_dir):
        result_file = os.path.join(task_dir, 'result.json')
        if os.path.exists(result_file):
            launch_result = json.loads(open(result_file).read())
            if launch_result['success']:
                return jsonify({
                    'success': True,
                    'task_status': 'done',
                })
            else:
                return jsonify({
                    'success': False,
                    'task_status': 'failed',
                    'failure_reason': launch_result['error_message']
                })
        else:
            return jsonify({
                'success': True,
                'task_status': 'failed',
                'failure_reason': 'ntest'
            })
    else:
        return jsonify({
            'success': False,
            'error_message': 'no such task'
        })


def get():
    return app


if __name__ == '__main__':
    app.run()
