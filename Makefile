.PHONY: help install test build arm-build arm-test

help:
	@echo "Targets:"
	@echo "  install   Install editable package (local)"
	@echo "  test      Run unit tests (local)"
	@echo "  build     Install + test (local)"
	@echo "  arm-build Build ARM container image via compose"
	@echo "  arm-test  Build + run tests in ARM container via compose"

install:
	python3 -m pip install --force-reinstall -e .

test:
	python3 -m unittest discover tests/

build: install test

arm-build:
	docker compose build

arm-test:
	docker compose up --build --abort-on-container-exit --exit-code-from arm-test
