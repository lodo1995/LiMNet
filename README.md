# LiMNet: Early-Stage Detection of IoT Botnets with Lightweight Memory Networks

### The paper

Giaretta, L., Lekssays, A., et al., _LiMNet: Early-Stage Detection of IoT Botnets with Lightweight Memory Networks_, 2021, in European Symposium on Research in Computer Security (ESORICS 2021)

### How to Run the Experiments

To run a single run of a single configuration (stored in a configuration file), using GPU number 0 (omit for CPU training) (output stored in `runs/limnet-best`):

```
python src/training.py limnet-best src/configs/medbiot-mem-gru-64.py --gpu 0
```

Example with a configuration file storing several named configurations, each repeated for several runs (output stored in several subfolders with pattern `multiruns/all-configs/<config-name>/trial_<N>`):

```
python src/train_all.py all-configs src/configs/all-configs.py --gpu 0
```

To get a summary of the size of a trained model (in this case using the output of a single configuration run):

```
python src/model_size.py runs/limnet-best
```

To measure the inference speed of a model on a single CPU core with batch size of 1 (reasons explained in the paper) (in this case using one particular run for one particular configuration from a multi-configuration, multi-run experiment):

```
python src/model_speed.py multiruns/all_configs/kitsune-lang-lstm-1-64-all_feats/trial_0
```

To produce Table 3 or Table 4 from the paper (requires `multiruns/all-configs` from the second example):

```
python src/make_latex_tables.py
```

### Dependencies

This code was tested in a Miniconda environment with the following libraries (and their dependencies):

```
cudatoolkit 11.1 (conda-forge)
ignite 0.5.0.dev20210314 (pytorch-nightly)
numpy 1.19.2
pandas 1.2.3
python 3.8.8
pytorch 1.9.0.dev20210329 (pytorch-nightly)
sklearn 0.24.1
scipy 1.6.2
tqdm 4.56.0
```
