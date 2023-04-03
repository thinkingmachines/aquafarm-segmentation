.PHONY: clean clean-test clean-pyc clean-build dev venv help
.DEFAULT_GOAL := help
-include .env

help:
	@awk -F ':.*?## ' '/^[a-zA-Z]/ && NF==2 {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

conda-env:
	conda env create -f environment.yml --no-default-packages

setup:
	conda install -c conda-forge gdal -y
	pip install pip-tools
	pip-sync requirements.txt
	pre-commit install

requirements:
	pip-compile requirements.in -o requirements.txt -v
	pip-sync requirements.txt

test:
	pytest -v tests/
