# LEPP

[**LEPP: Language-conditioned Equivariant Pick & Place**](https://google.com)
Mingxi Jia*, Haojie Huang*, Robert Platt, Stefanie Tellex

\* Equal contribution

## Installation
1. create conda env
```
conda create -n LEPP python==3.8.10
conda activate LEPP
```
2. install dependencies
```
# tested on ubuntu20.04, cuda11.8, python3.8.10
# conda install pytorch 2.1
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# install other dependencies
pip install -r requirements.txt
# export env variable
export PYTHONPATH=/YOURPATH_TO_REPO:$PYTHONPATH
export CLIPORT_ROOT=$(pwd)
```
3. run baseline (cliport) training and eval
```
python cliport/train_new.py train.task=stack-block-pyramid-seq-seen-colors train.agent=cliport train.n_demos=100  train.n_steps=200000 train.save_freq_step=10000 train.exp_folder=exps  dataset.cache=False

python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors agent=cliport mode=val n_demos=100 train_demos=100 checkpoint_type=val_missing
```
4. run our method
```
# LEPP-clip-similarity-input (res50)
python cliport/train_new.py train.task=stack-block-pyramid-seq-seen-colors train.agent=LEPP lepp.model_name=cliport-similarity-head lepp.pick_kernel_name=unetl lepp.place_kernel_name=unet train.n_demos=100  train.n_steps=15000  train.exp_folder=exps  dataset.cache=False

python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors agent=LEPP lepp.model_name=cliport-similarity-head lepp.pick_kernel_name=unetl lepp.place_kernel_name=unet mode=val n_demos=100 train_demos=100 checkpoint_type=val_missing

# LEPP-clip-similarity-input (vit32)
python cliport/train_new.py train.task=stack-block-pyramid-seq-seen-colors train.agent=LEPP lepp.model_name=cliport-similarity-head-hugging lepp.pick_kernel_name=unetl lepp.place_kernel_name=unet train.n_demos=100  train.n_steps=15000 train.save_freq_step=1000  train.exp_folder=exps  dataset.cache=False

python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors agent=LEPP lepp.model_name=cliport-similarity-head-hugging lepp.pick_kernel_name=unetl lepp.place_kernel_name=unet mode=val n_demos=100 train_demos=100 checkpoint_type=val_missing

```