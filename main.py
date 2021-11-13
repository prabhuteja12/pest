import sys

import pytorch_lightning as pl
import torch

from adaptation_methods import get_adaptation_method
from dataset import get_dataset
from dataset.wrapper import get_dataset
from networks import get_model
from utils import TensorboardSummary, load_config, make_summary_folder
from utils.performance import accuracy_clean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cfg = None
pl.seed_everything(12, True)


def main(cfg):
    model = get_model(cfg.dataset)
    dataset = get_dataset(
        cfg.dataset, augmentation=cfg.augmentation, corruption=cfg.corruption, level=cfg.level, **cfg.augmentation_params
    )

    baseline_acc = accuracy_clean(
        get_adaptation_method("baseline")(model), dataset, cfg.train_batch_size, device=device, class_subset=None
    )
    
    wrapper_model = get_adaptation_method(cfg.adapt_method)(
        model=model,
        optimizer=cfg.optimizer,
        num_steps=cfg.num_steps,
        freeze_bn=cfg.freeze_bn,
        freeze_classifier=cfg.freeze_classifier,
        reg=cfg.reg,
        optim_parameters=cfg.optim_parameters,
        episodic=cfg.episodic, 
        loss_fn=cfg.loss,
        loss_lambda=cfg.loss_lambda
    )
    acc = accuracy_clean(wrapper_model, dataset, cfg.train_batch_size, device=device, class_subset=None)
    summary_writer = TensorboardSummary(make_summary_folder(cfg))
    summary_writer.save_experiment_config(cfg, {"baseline": 100 * baseline_acc, "final": 100 * acc})
    
    print(f'Intial: {baseline_acc * 100}, Final: {100 * acc}')


if __name__ == "__main__":
    config_file = sys.argv[1]
    cfg = load_config(config_file)
    print(cfg)
    main(cfg)
