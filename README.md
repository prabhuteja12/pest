# Code for Test time Adaptation through PErturbation robuSTness (PEST) 
Code release for paper [**Test time Adaptation through Perturbation Robustness**](https://openreview.net/forum?id=GbBeI5z86uD)
by [Prabhu Teja S](https://prabhuteja12.github.io/), and [Fran&ccedil;ois Fleuret](https://fleuret.org/francois/) to be published at NeurIPS 2021 Workshop on
Distribution Shifts.



## Installation

The requirements are `pytorch`, `pytorch-lightning`, and [`RobustBench`](https://github.com/RobustBench/robustbench) package. 


## Running the code

The code heavily relies on yaml formatted configuration files. These are read in [`main.py`](main.py) and passed onto the
appropriate function. A [sample configuration](sample_config.yaml) file is given. The code can be run with 
```python 
python main.py sample_config.yaml
```

## Downloading the datasets
The current codebase supports [CIFAR-10-C](https://zenodo.org/record/2535967), [CIFAR-100-C](https://zenodo.org/record/3555552), and [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification
) datasets. They have to be downloaded and placed in
the datasets folder.


## Citing

If you find me work or code useful, consider citing us using

```bibtex
@misc{sivaprasad2021test,
      title={Test time Adaptation through Perturbation Robustness}, 
      author={Prabhu Teja Sivaprasad and François Fleuret},
      year={2021},
      eprint={2110.10232},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
