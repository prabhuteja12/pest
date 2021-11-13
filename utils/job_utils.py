from tools.general import launch_multi_job


def launch_job_configs(start_index, stop_index, folder="./cfs", group_size=8, ):
    commands = [f"python main.py {folder}/{x}.yaml" for x in range(start_index, stop_index + 1)]
    sub_list = [commands[n : n + group_size] for n in range(0, len(commands), group_size)] 

    for sl in sub_list:
        launch_multi_job.submit_multijob(sl, gpu="sgpu", conda="pyt18", extramem=False, extraflags=r'-l "h=vgn[gfhi]*"')

