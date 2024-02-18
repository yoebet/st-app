def get_task_dir(tasks_dir: str, task_id: str, sub_dir: str = None):
    if sub_dir == '' or sub_dir == '_':
        sub_dir = None
    if sub_dir is not None:
        task_dir = f'{tasks_dir}/{sub_dir}/t_{task_id}'
    else:
        task_dir = f'{tasks_dir}/t_{task_id}'

    return task_dir


def get_tf_logging_dir(TF_LOGS_DIR: str, sub_dir: str = None):
    logging_dir = TF_LOGS_DIR
    if sub_dir == '' or sub_dir == '_':
        sub_dir = None
    if sub_dir is not None:
        return f'{logging_dir}/{sub_dir}'
    return logging_dir
