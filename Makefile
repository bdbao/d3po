.PHONY: all install utils unzip sample

all: install utils unzip sample

install:
  git clone https://github.com/bdbao/d3po
  cd d3po && git checkout add-inpaint && pip install -e .

utils:
  pip3 install git+https://github.com/XuehaiPan/nvitop.git#egg=nvitop
  apt-get install unzip
  nvitop

unzip:
  # Upload train_data.zip
  unzip train_data.zip

sample:
  accelerate launch scripts/sample.py
