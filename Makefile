.PHONY: install develop build build-legacy prepare-data train-cuda train-torch test info smoke

install:
	python -m pip install -e .

develop:
	python -m pip install -e .[torch,dev]

build:
	minicnn build

build-legacy:
	minicnn build --legacy-make

prepare-data:
	minicnn prepare-data

train-cuda:
	minicnn train-cuda --config configs/train_cuda.yaml

train-torch:
	minicnn train-torch --config configs/train_torch.yaml

test:
	pytest -q

info:
	minicnn info

smoke:
	minicnn smoke
