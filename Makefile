all: install

install:
	pip install .

lint:
	flake8

