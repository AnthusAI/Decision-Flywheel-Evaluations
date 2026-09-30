.PHONY: test download preflight select run report install-tools release

PYTHON ?= python
STAGE ?= scoreboard
MAX_RETRIES_PER_REQUEST ?= 0
DOWNLOAD_CACHE ?= .data/huggingface

ifeq ($(strip $(ROWS)),)
SOURCE_ARGS = --dataset-cache "$(DATASET_CACHE)"
else
SOURCE_ARGS = --rows "$(ROWS)"
endif

test:
	$(PYTHON) -m pytest -q

# Only this explicit setup target can acquire datasets; no model is constructed.
download:
	$(PYTHON) scripts/download_pinned_datasets.py download --cache-root "$(DOWNLOAD_CACHE)" $(CONFIRM)

# With missing variables these print usage and stop before data or a provider.
# Inputs are intentionally explicit and are never downloaded by these targets.
preflight:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || test -z "$(OUTPUT)" || { test -z "$(ROWS)" && test -z "$(DATASET_CACHE)"; } || { test -n "$(ROWS)" && test -n "$(DATASET_CACHE)"; }; then \
		echo "Usage: make preflight PROTOCOL=... MANIFEST=... (ROWS=fixture.json | DATASET_CACHE=.data/huggingface) OUTPUT=... [STAGE=scoreboard]"; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.cli preflight --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" $(SOURCE_ARGS) --output "$(OUTPUT)" --stage "$(STAGE)"

select:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || { test -z "$(ROWS)" && test -z "$(DATASET_CACHE)"; } || { test -n "$(ROWS)" && test -n "$(DATASET_CACHE)"; } || test -z "$(PREFLIGHT)" || test -z "$(LEDGER)" || test -z "$(PREREGISTRATION)" || test -z "$(MODEL_IDENTITY)" || test -z "$(PROVIDER_MODEL)" || test -z "$(ATTEMPT_CEILING)" || test -z "$(ARTIFACT_REFERENCE)" || test -z "$(PROTOCOL_OUTPUT)" || test -z "$(ARTIFACT_OUTPUT)" || test -z "$(REGISTRATION_OUTPUT)"; then \
		echo "Usage: make select PROTOCOL=... MANIFEST=... (ROWS=fixture.json | DATASET_CACHE=.data/huggingface) PREFLIGHT=... LEDGER=... PREREGISTRATION=... MODEL_IDENTITY=... PROVIDER_MODEL=... ATTEMPT_CEILING=... ARTIFACT_REFERENCE=... PROTOCOL_OUTPUT=... ARTIFACT_OUTPUT=... REGISTRATION_OUTPUT=..."; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.cli select --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" $(SOURCE_ARGS) --preflight "$(PREFLIGHT)" --ledger "$(LEDGER)" --preregistration "$(PREREGISTRATION)" --model-identity "$(MODEL_IDENTITY)" --provider-model "$(PROVIDER_MODEL)" --attempt-ceiling "$(ATTEMPT_CEILING)" --artifact-reference "$(ARTIFACT_REFERENCE)" --protocol-output "$(PROTOCOL_OUTPUT)" --artifact-output "$(ARTIFACT_OUTPUT)" --registration-output "$(REGISTRATION_OUTPUT)"

run:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || { test -z "$(ROWS)" && test -z "$(DATASET_CACHE)"; } || { test -n "$(ROWS)" && test -n "$(DATASET_CACHE)"; } || test -z "$(PREFLIGHT)" || test -z "$(LEDGER)" || test -z "$(PREREGISTRATION)" || test -z "$(OUTPUT)" || test -z "$(PROVIDER_MODEL)" || test -z "$(ATTEMPT_CEILING)" || test -z "$(MAX_NEW)" || test -z "$(BASE_URL)" || test -z "$(TIMEOUT_SECONDS)" || test -z "$(ADAPTER_REVISION)" || test -z "$(PACKAGE_REVISION)"; then \
		echo "Usage: make run PROTOCOL=... MANIFEST=... (ROWS=fixture.json | DATASET_CACHE=.data/huggingface) PREFLIGHT=... LEDGER=... PREREGISTRATION=... OUTPUT=... PROVIDER_MODEL=... BASE_URL=https://... TIMEOUT_SECONDS=... ADAPTER_REVISION=... PACKAGE_REVISION=typesafe-sdk-0.7.1 ATTEMPT_CEILING=... MAX_NEW=... CONFIRM=--confirm"; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.cli run --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" $(SOURCE_ARGS) --preflight "$(PREFLIGHT)" --ledger "$(LEDGER)" --preregistration "$(PREREGISTRATION)" --output "$(OUTPUT)" --provider-model "$(PROVIDER_MODEL)" --base-url "$(BASE_URL)" --timeout-seconds "$(TIMEOUT_SECONDS)" --adapter-revision "$(ADAPTER_REVISION)" --package-revision "$(PACKAGE_REVISION)" --attempt-ceiling "$(ATTEMPT_CEILING)" --max-new "$(MAX_NEW)" --max-retries-per-request "$(MAX_RETRIES_PER_REQUEST)" $(RETRY_FAILED) $(RECOVER_UNCERTAIN) "$(CONFIRM)"

report:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || test -z "$(PREFLIGHT)" || test -z "$(OBSERVATIONS)" || test -z "$(OUTPUT)"; then \
		echo "Usage: make report PROTOCOL=... MANIFEST=... PREFLIGHT=... OBSERVATIONS=... OUTPUT=..."; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.cli report --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" --preflight "$(PREFLIGHT)" --observations "$(OBSERVATIONS)" --output "$(OUTPUT)"

install-tools:
	$(PYTHON) -m pip install -e '.[tools]'

release:
	$(PYTHON) -m semantic_release version
