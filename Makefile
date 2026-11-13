.PHONY: help test fixture run run-demo lambda-check clean

SERVICE_NAME ?= stim-events
PYTHON ?= python3
IMAGE ?= stim-events:dev
DEMO ?= ../workflow-physiology/data/work/d349/physio-import

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make test         - pytest: synthetic physio-import output through the processor (no Docker)"
	@echo "make fixture      - write the synthetic fixture to data/fixture/ (physio/, params/, truth.json)"
	@echo "make run          - build the image and run it in Docker over data/fixture/, writing to data/output/"
	@echo "make run-demo     - run the image over DEMO=$(DEMO) into data/output-demo/ (--memory 2g)"
	@echo "make lambda-check - run the image as Lambda runs it: read-only, any uid, HOME/TMPDIR under /tmp"
	@echo "make clean        - remove data/output*/ and data/fixture/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

test:
	$(PYTHON) -m pytest -q tests

fixture:
	$(PYTHON) -m tests.make_fixture data/fixture

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -type f | sort

run-demo:
	docker build --platform linux/amd64 -t $(IMAGE) .
	rm -rf data/output-demo && mkdir -p data/output-demo
	docker run --rm --platform linux/amd64 --cpus 1 --memory 2g --env-file dev.env \
		-v "$(abspath $(DEMO))":/data/input:ro -v "$(CURDIR)/data/output-demo":/data/output $(IMAGE)

lambda-check: fixture
	docker build --platform linux/amd64 -t $(IMAGE) .
	rm -rf data/output-lambda && mkdir -p data/output-lambda && chmod 777 data/output-lambda
	docker run --rm --platform linux/amd64 --memory 2g --read-only --user 65534:65534 --tmpfs /tmp:mode=1777 \
		-e HOME=/tmp/w/home -e TMPDIR=/tmp/w/tmp -e INPUT_DIR=/tmp/s/input -e OUTPUT_DIR=/tmp/s/output \
		-v "$(CURDIR)/data/fixture":/tmp/s/input:ro -v "$(CURDIR)/data/output-lambda":/tmp/s/output $(IMAGE)

clean:
	rm -rf data/output data/output-demo data/output-lambda data/fixture
