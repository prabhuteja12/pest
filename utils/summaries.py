import traceback
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch.utils.tensorboard import SummaryWriter


class TensorboardSummary(SummaryWriter):
    def __init__(self, save_folder):
        self.directory = Path(save_folder)
        self.runs = sorted(
            self.directory.glob("experiment_*"),
            key=lambda x: int(str(x).split("_")[-1]),
        )

        run_id = int(str(self.runs[-1]).split("_")[-1]) + 1 if self.runs else 0
        self.experiment_dir = Path(self.directory / "experiment_{}".format(run_id))
        self.experiment_dir.mkdir(exist_ok=True, parents=True)
        super().__init__(str(self.experiment_dir))

    def write_metrics(self, metrics, epoch):
        
        for name, value in metrics.items():
            self.add_scalar("val/%s" % name, value, epoch)

    def save_experiment_config(self, args, perf=None):
        config = {}
        args = pd.json_normalize(dict(args), sep='_').to_dict(orient='records')[0]
        for k, v in args.items():
            config[k] = str(v)
        perf = perf if perf else {}
        self.add_hparams(config, perf)


def make_summary_folder(cfg):
    log_dir = cfg.log_dir
    method = cfg.adapt_method
    augmentation = cfg.augmentation
    dataset = cfg.dataset
    
    prefix = f'_{cfg.prefix}' if cfg.prefix else ''

    if dataset in ['cifar10', 'cifar100', 'imagenet']:
        dataset = f"{dataset}_{cfg.level}"
        
    folder_structure = Path(log_dir) / dataset / augmentation / f"{method}{prefix}"
    folder_structure.mkdir(parents=True, exist_ok=True)

    return folder_structure.as_posix()


def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        print(path)
        event_acc.Reload()
        tags = event_acc.Tags()#["scalars"]
        print(tags)
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def read_logs(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs
