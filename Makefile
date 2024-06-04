.PHONY: all install utils unzip sample train

all: install utils unzip sample train

install:
    git clone https://github.com/bdbao/d3po
    cd d3po && pip install -e .

utils:
    pip3 install git+https://github.com/XuehaiPan/nvitop.git#egg=nvitop
    apt-get install unzip
    nvitop

unzip:
    # Upload train_data.zip
    unzip train_data.zip

sample:
    accelerate launch scripts/sample_inpaint.py

train:
    accelerate launch scripts/train_inpaint.py