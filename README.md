# Constrained-Labeling-for-Weakly-Supervised-Learning
This repo contains code for Constrained Labeling for Weakly Supervised Learning

If you use this work in an academic study, please cite our paper

```
@article{arachie2020constrained,
  title={Constrained Labeling for Weakly Supervised Learning},
  author={Arachie, Chidubem and Huang, Bert},
  journal={arXiv preprint arXiv:2009.07360},
  year={2020}
}
```

# Requirements

The library is tested in Python 3.6 and 3.7. Its main requirement is numpy, Tensorflow is also required to train generated labels


# Algorithm

The most important script is the train_CLL.py script that contains implementation of the algorithm. The other scripts are secondary classes for running experiments

# Examples

The file run_experiments creates synthetic example for the experiment in the paper. It also runs the real data experiments.

To run examples on real datasets from the paper; 1. download the datasets, 2. run generate_weak_signals, and 3. use run_experiment script in the run_experiments file

