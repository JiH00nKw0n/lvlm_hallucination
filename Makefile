IMAGE_NAME := evaluate-lvlm-hallucination-mitigators
CFG_PATH := config/evaluate/llama3-llava-next-8b-hf.yaml
LOG_DIR := .log
CACHE_DIR := .cache
HF_TOKEN ?= $(HFtoken)

.PHONY: build-evaluate-lvlm-hallucination-mitigators run-evaluate-lvlm-hallucination-mitigators

build-evaluate-lvlm-hallucination-mitigators:
	docker build -t $(IMAGE_NAME) .

run-evaluate-lvlm-hallucination-mitigators:
	@mkdir -p $(LOG_DIR) $(CACHE_DIR)
	@LOG_FILE="$(LOG_DIR)/eval_$$(date +%Y%m%d_%H%M%S).log"; \
	echo "Logging to: $$LOG_FILE"; \
	docker run --rm -it \
		--gpus all \
		-v "$$(pwd)":/workspace \
		-v "$$(pwd)/$(LOG_DIR)":/workspace/.log \
		-v "$$(pwd)/$(CACHE_DIR)":/workspace/.cache \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e HF_HOME=/workspace/.cache \
		-e HF_DATASETS_CACHE=/workspace/.cache \
		-e LOG_DIR=/workspace/.log \
		$(IMAGE_NAME) \
		bash -lc "cd /workspace && python evaluate.py --cfg-path $(CFG_PATH) 2>&1 | tee -a /workspace/$$LOG_FILE"
