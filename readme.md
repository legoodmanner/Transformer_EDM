# Transformer_EDM 
![](https://i.imgur.com/AW7BlnL.png)
## Introduction
This is the repo regarding experiments of EDM music generation using VQ-VAE Transformer. We experimented with different lengths of token sequences and showed the relationship between validity and sequence length in the fixed BPM of 120.

## Hyperparameters:
* `data_path`: The path of the trainig and testing data directory.
* `checkpoint_dir`: The directory path of VQ model parameter files.
* `transformer_checkpoint_dir`: The directory path of language model parameter files.
* `sample_dir`: The directory path placing VQ reconstructed samples.
* `transformer_sample_dir`: The directory path placing language model testing samples.
* `save_file_path`: The directory path placing generated samples.
* `n_embed`: Vocabulary size of the codebook
* `encoder_scale_factors`: The sequence of scaled ratios of the encoder. The sequence length would be `256 * np.prod(encoder_scale_factors)`


## Usage
### **Step 1** 
Before training VQ-VAE, hyperparameters such as `data_path`, `checkpoint_dir`, `sample_dir`, `n_embed` need to be revised to valid path.
```
bash train_vqvae.sh
```
### **Step 2**
Again, Before training the language model, `data_path`, `checkpoint_dir`, `sample_dir`, `transformer_checkpoint_dir`, `transformer_sample_dir`, `n_embed` should be reviced to valid path. 
```
bash train_transformer.sh
```
### **Step 3**
To generate the loop sample, please run:
```
bash generate.sh
```
Please determined the hyperparameters of `checkpoint_dir`
,`transformer_checkpoint_dir`, `save_file_path`, `data_path`, `n_embed` beforehand.
## Dataset
All the clips included in loop dataset are from [Freesound.org](https://freesound.org/).
## Experiment
The name of the parameter set would be named:
```
{n_embed}_{embed_dim}_{sequence_length}
```
* **Inception score** is obtained by a pre-trained classifier trained on Looperman dataset.
* **Fréchet Audio Distance** is evaluated by [1]

| Name        | Inception Score (IS) | Fréchet Audio Distance (FAD) |
|:----------- |:-------------------- |:---------------------------- |
| `64_80_16`  | 3.97 +- 1.91         | 3.41                         |
| `64_80_20`  | 2.11 +- 2.03         | 7.22                         |
| `64_80_32`  | 3.71 +- 2.12         | 4.12                         |
| `64_80_60`  | 2.29 +- 1.95         | 8.91                         |
| `64_80_256` | 3.21 +- 2.55         | 4.91                         |


| Name        | Inception Score (IS) | Fréchet Audio Distance (FAD) |
| :---------- | :------------        | :----                        |
| `64_80_16`  | 3.97 +- 1.91         | 3.41                         |
| `32_80_16`  | 4.21 +- 1.64         | 4.01                         |
| `16_80_16`  | 3.59 +- 2.00         | 2.91                         |

| Name        | Inception Score (IS) | Fréchet Audio Distance (FAD) |
|:----------- |:-------------------- |:---------------------------- |
| `64_80_16`  | 3.97 +- 1.91         | 3.41                         |
| `64_40_16`  | 3.31 +- 1.84         | 3.20                         |
| `64_160_16` | 4.09 +- 2.30         | 5.31                         |

* Throgh the experiments, **sequence length** is the most significant factor in the objective evaluation when generating loops of indentical BPM of 120.
* If the songs in the training data is all 4/4 beat or 2/4 beat. It is recommended to have the sequence length with the number of power of 2.
* Some generated demo samples are provided in the directory "demo".

## Reference
[1] Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, Matthew Sharifi. "Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms". 2018



