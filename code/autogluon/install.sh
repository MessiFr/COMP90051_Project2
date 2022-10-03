conda create -y --force -n ag python3.8 pip
conda activate ag

pip install "mxnet<2.0.0"
pip install autogluon