## Enviroment

**NOTE: a misaligned enviroment between inference and training may cause bad effect.**

**Feel free to raise issue if there is any enviroment problem!**

- create and activate env
```shell
conda create -n rscovlm python=3.10.12
conda activate rscovlm
```

- set cuda&gcc (recommanded for current enviroment, you can also set it in ~/.bashrc)
```shell
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
touch $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
vim $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
```
write the following lines
```shell
# set cuda&gcc home
export CUDA_HOME=/mnt/petrelfs/share_data/liqingyun/cuda/cuda-12.4/  # change this to <path to cuda-12.1>
export GCC_HOME=/mnt/petrelfs/share/gcc-10.1.0/  # change this to <path to gcc (such as 10.1)>
# remove redundant cuda&gcc path
export PATH=$(echo "$PATH" | sed -e 's#[^:]*cuda[^:]*:##g' -e 's#:[^:]*cuda[^:]*##g' -e 's#[^:]*gcc[^:]*:##g' -e 's#:[^:]*gcc[^:]*##g')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's#[^:]*cuda[^:]*:##g' -e 's#:[^:]*cuda[^:]*##g' -e 's#[^:]*gcc[^:]*:##g' -e 's#:[^:]*gcc[^:]*##g')
# set cuda&gcc path
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$GCC_HOME/lib64:$LD_LIBRARY_PATH
# set site-packages path
export SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
```
then `conda activate rscovlm` to enable these env vars

- install torch
```shell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

- install transformers and peft
```shell
pip install transformers==4.51.3 peft==0.14.0
```

- install vllm (optional)
```shell
pip install vllm==0.7.3
# If xgrammar raise installation error, we should build xgrammar first
cd $SITE_PACKAGES_PATH
pip install ninja pybind11
git clone --recursive https://github.com/mlc-ai/xgrammar.git && cd xgrammar
git checkout v0.1.11
mkdir build && cd build
conda install cmake # need new version of cmake
cmake .. -G Ninja # remove -Werror from CMakeLists.txt
ninja
cd ../python
pip install .
pip install vllm==0.7.3
```

- build and install [mmcv](https://mmcv.readthedocs.io/en/latest/)
```shell
# install mim
pip install openmim

# install mmcv (three manners)
# install from branch forked by Li-Qingyun (recommanded)
git clone https://github.com/Li-Qingyun/mmcv.git $SITE_PACKAGES_PATH/mmcv
cd $SITE_PACKAGES_PATH/mmcv
git checkout v2.0.1-lqy-fix
bash install.sh
# install with openmim
mim install "mmcv==2.0.1"
# install from source
git clone https://github.com/open-mmlab/mmcv.git $SITE_PACKAGES_PATH/mmcv
cd $SITE_PACKAGES_PATH/mmcv
git checkout v2.0.1
pip install -r requirements/optional.txt
echo 'set -x;TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print(f'\''{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}'\'')") pip install -e . -v' >> install.sh
bash install.sh
```
The compiling of mmcv-v2.0.1 may raise error, because torch require C++17 or later compatible compiler. One solution is in [this issue](https://github.com/open-mmlab/mmcv/issues/2860).
> Changing `c++14` to `c++17` in [the 204 line](https://github.com/open-mmlab/mmcv/blob/d28aa8a9cced3158e724585d5e6839947ca5c449/setup.py#L204) and [the 421 line](https://github.com/open-mmlab/mmcv/blob/d28aa8a9cced3158e724585d5e6839947ca5c449/setup.py#L421) of the `setup.py` can temporarily fix this issue.

- install openmmlab mmdet and mmrotate
```shell
mim install "mmdet==3.0.0"
mim install "mmrotate==1.0.0rc1"
```

- build and install [mmcv](https://mmcv.readthedocs.io/en/latest/)
```shell
# install mim
pip install openmim

# install mmcv
# install from branch forked by Li-Qingyun (recommanded)
git clone https://github.com/Li-Qingyun/mmcv.git $SITE_PACKAGES_PATH/mmcv
cd $SITE_PACKAGES_PATH/mmcv
git checkout v2.0.1-lqy-fix
bash install.sh
# install with openmim
mim install "mmcv==2.0.1"
# install from source 
git clone https://github.com/open-mmlab/mmcv.git $SITE_PACKAGES_PATH/mmcv
cd $SITE_PACKAGES_PATH/mmcv
git checkout v2.0.1
pip install -r requirements/optional.txt
echo 'set -x;TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print(f'\''{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}'\'')") pip install -e . -v' >> install.sh
bash install.sh  # you may need to read and follow the note below first
cd -
```
The compiling of mmcv-v2.0.1 may raise error, because torch require C++17 or later compatible compiler. One solution is in [this issue](https://github.com/open-mmlab/mmcv/issues/2860).
> Changing `c++14` to `c++17` in [the 204 line](https://github.com/open-mmlab/mmcv/blob/d28aa8a9cced3158e724585d5e6839947ca5c449/setup.py#L204) and [the 421 line](https://github.com/open-mmlab/mmcv/blob/d28aa8a9cced3158e724585d5e6839947ca5c449/setup.py#L421) of the `setup.py` can temporarily fix this issue.

- install openmmlab mmdet and mmrotate
```shell
mim install "mmdet==3.0.0"
mim install "mmrotate==1.0.0rc1"
```

- install [flash-attention](https://github.com/Dao-AILab/flash-attention) and [liger-kernel](https://github.com/linkedin/Liger-Kernel)
```shell
pip install flash-attn==2.7.0.post2 --no-build-isolation
pip install liger-kernel==0.5.4

- install av
```
conda install -c conda-forge av
```

- install rscovlm
```shell
pip install -e .
```

- install VLMEvalKit (optional)
```shell
git clone https://github.com/open-compass/VLMEvalKit.git $SITE_PACKAGES_PATH/VLMEvalKit
cd $SITE_PACKAGES_PATH/VLMEvalKit
git checkout c59d4d8  # optional (to be updated)
pip install -e .
```

- install lmms-eval (optional) 
> NOTE that the requirements of lmms-eval may break your enviroment
```shell
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval $SITE_PACKAGES_PATH/lmms-eval
cd $SITE_PACKAGES_PATH/lmms-eval
git checkout 2714c46  # optional (to be updated)
pip install -e .
```

- install ms-swift (optional)
```
pip install -U ms-swift
```

- install xtuner (optional)
```
pip install -U xtuner
```
