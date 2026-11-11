# Common tasks. Windows users: run the commands inside each target manually (or use WSL).
PY ?= python

.PHONY: install data synthetic train-baseline train train-edge ablation evaluate export api web test docker

install:            ## Python + Node dependencies
	$(PY) -m pip install -e "ml[train,dev]" -r backend/requirements-dev.txt
	cd frontend && npm install

data:               ## Download the NIH dataset (~337 MB)
	cd ml && $(PY) scripts/download_data.py --out data

synthetic:          ## Tiny fake dataset for smoke tests
	cd ml && $(PY) scripts/make_synthetic_data.py --out data/synthetic

train-baseline:     ## From-scratch CNN (patient split)
	cd ml && $(PY) -m malariascope.train --config configs/baseline_cnn.yaml --output-dir ../artifacts

train:              ## Main model: EfficientNet-B0 transfer learning (patient split)
	cd ml && $(PY) -m malariascope.train --config configs/efficientnet_b0.yaml --output-dir ../artifacts

train-edge:         ## MobileNetV3-Small for TFLite / phones
	cd ml && $(PY) -m malariascope.train --config configs/mobilenet_v3_small.yaml --output-dir ../artifacts

ablation:           ## Same model with a leaky cell-level split (to quantify leakage)
	cd ml && $(PY) -m malariascope.train --config configs/ablation_cell_split.yaml --output-dir ../artifacts/ablation

export:             ## TFLite fp32/fp16/int8 + latency/size/agreement report
	cd ml && $(PY) -m malariascope.export --model-dir ../artifacts/latest --run-dir $$(ls -td ../artifacts/runs/*/ | head -1)

api:                ## FastAPI on :8000 (docs at /docs)
	cd backend && uvicorn app.main:app --reload --port 8000

web:                ## React dev server on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

test:
	cd ml && $(PY) -m pytest -q
	cd backend && $(PY) -m pytest -q

docker:
	docker compose up --build
