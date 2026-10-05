.PHONY: reviews-stream-fixture reviews-stream-report unified-flywheel-test unified-flywheel-dry-run test download preflight select run report pilot-preflight pilot pilot-report ordering-preflight ordering-run ordering-report install-tools release

PYTHON ?= python
STAGE ?= scoreboard
MAX_RETRIES_PER_REQUEST ?= 0
MAX_CONCURRENCY ?= 1
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
	$(PYTHON) -m decision_flywheel_evaluations.cli run --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" $(SOURCE_ARGS) --preflight "$(PREFLIGHT)" --ledger "$(LEDGER)" --preregistration "$(PREREGISTRATION)" --output "$(OUTPUT)" --provider-model "$(PROVIDER_MODEL)" --base-url "$(BASE_URL)" --timeout-seconds "$(TIMEOUT_SECONDS)" --adapter-revision "$(ADAPTER_REVISION)" --package-revision "$(PACKAGE_REVISION)" --attempt-ceiling "$(ATTEMPT_CEILING)" --max-new "$(MAX_NEW)" --max-retries-per-request "$(MAX_RETRIES_PER_REQUEST)" --max-concurrency "$(MAX_CONCURRENCY)" $(RETRY_FAILED) $(RECOVER_UNCERTAIN) "$(CONFIRM)"

report:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || test -z "$(PREFLIGHT)" || test -z "$(OBSERVATIONS)" || test -z "$(OUTPUT)"; then \
		echo "Usage: make report PROTOCOL=... MANIFEST=... PREFLIGHT=... OBSERVATIONS=... OUTPUT=..."; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.cli report --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" --preflight "$(PREFLIGHT)" --observations "$(OBSERVATIONS)" --output "$(OUTPUT)"

pilot-preflight:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || test -z "$(PREFLIGHT)" || test -z "$(OUTPUT)" || { test -z "$(ROWS)" && test -z "$(DATASET_CACHE)"; } || { test -n "$(ROWS)" && test -n "$(DATASET_CACHE)"; }; then \
		echo "Usage: make pilot-preflight PROTOCOL=... MANIFEST=... PREFLIGHT=... (ROWS=fixture.json | DATASET_CACHE=.data/huggingface) OUTPUT=... [OVERWRITE=--overwrite]"; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.pilot_cli preflight --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" --source-preflight "$(PREFLIGHT)" $(SOURCE_ARGS) --output "$(OUTPUT)" $(OVERWRITE)

pilot:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || test -z "$(PREFLIGHT)" || test -z "$(PILOT)" || test -z "$(LEDGER)" || test -z "$(PREREGISTRATION)" || test -z "$(OUTPUT)" || test -z "$(PROVIDER_MODEL)" || test -z "$(BASE_URL)" || test -z "$(TIMEOUT_SECONDS)" || test -z "$(ADAPTER_REVISION)" || test -z "$(PACKAGE_REVISION)" || test -z "$(ATTEMPT_CEILING)" || test -z "$(MAX_NEW)" || { test -z "$(ROWS)" && test -z "$(DATASET_CACHE)"; } || { test -n "$(ROWS)" && test -n "$(DATASET_CACHE)"; }; then \
		echo "Usage: make pilot PROTOCOL=... MANIFEST=... PREFLIGHT=... PILOT=... (ROWS=fixture.json | DATASET_CACHE=.data/huggingface) LEDGER=... PREREGISTRATION=... OUTPUT=... PROVIDER_MODEL=... BASE_URL=https://... TIMEOUT_SECONDS=... ADAPTER_REVISION=... PACKAGE_REVISION=typesafe-sdk-0.7.1 ATTEMPT_CEILING=... MAX_NEW=... CONFIRM=--confirm"; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.pilot_cli run --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" --source-preflight "$(PREFLIGHT)" --pilot "$(PILOT)" $(SOURCE_ARGS) --ledger "$(LEDGER)" --preregistration "$(PREREGISTRATION)" --output "$(OUTPUT)" --provider-model "$(PROVIDER_MODEL)" --base-url "$(BASE_URL)" --timeout-seconds "$(TIMEOUT_SECONDS)" --adapter-revision "$(ADAPTER_REVISION)" --package-revision "$(PACKAGE_REVISION)" --attempt-ceiling "$(ATTEMPT_CEILING)" --max-new "$(MAX_NEW)" $(CONFIRM) $(OVERWRITE)

pilot-report:
	@if test -z "$(PROTOCOL)" || test -z "$(PILOT)" || test -z "$(OBSERVATIONS)" || test -z "$(OUTPUT)"; then \
		echo "Usage: make pilot-report PROTOCOL=... PILOT=... OBSERVATIONS=... OUTPUT=..."; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.pilot_cli report --protocol "$(PROTOCOL)" --pilot "$(PILOT)" --observations "$(OBSERVATIONS)" --output "$(OUTPUT)" $(OVERWRITE)

ordering-preflight:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || test -z "$(PREFLIGHT)" || test -z "$(INITIAL_OBSERVATIONS)" || test -z "$(RESULT_REFERENCE)" || test -z "$(ORDER_SEEDS)" || test -z "$(OUTPUT)" || { test -z "$(ROWS)" && test -z "$(DATASET_CACHE)"; } || { test -n "$(ROWS)" && test -n "$(DATASET_CACHE)"; }; then \
		echo "Usage: make ordering-preflight PROTOCOL=... MANIFEST=... PREFLIGHT=initial.preflight.json INITIAL_OBSERVATIONS=... RESULT_REFERENCE=results/initial.json ORDER_SEEDS='0 1 2 3 4' (ROWS=fixture.json | DATASET_CACHE=.data/huggingface) OUTPUT=... [OVERWRITE=--overwrite]"; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.ordering_cli preflight --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" --initial-preflight "$(PREFLIGHT)" --initial-observations "$(INITIAL_OBSERVATIONS)" --initial-result-reference "$(RESULT_REFERENCE)" $(foreach seed,$(ORDER_SEEDS),--shuffle-seed "$(seed)") $(SOURCE_ARGS) --output "$(OUTPUT)" $(OVERWRITE)

ordering-run:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || test -z "$(PREFLIGHT)" || test -z "$(INITIAL_OBSERVATIONS)" || test -z "$(ORDERING)" || test -z "$(LEDGER)" || test -z "$(PREREGISTRATION)" || test -z "$(OUTPUT)" || test -z "$(PROVIDER_MODEL)" || test -z "$(BASE_URL)" || test -z "$(TIMEOUT_SECONDS)" || test -z "$(ADAPTER_REVISION)" || test -z "$(PACKAGE_REVISION)" || test -z "$(ATTEMPT_CEILING)" || test -z "$(MAX_NEW)" || { test -z "$(ROWS)" && test -z "$(DATASET_CACHE)"; } || { test -n "$(ROWS)" && test -n "$(DATASET_CACHE)"; }; then \
		echo "Usage: make ordering-run PROTOCOL=... MANIFEST=... PREFLIGHT=initial.preflight.json INITIAL_OBSERVATIONS=... ORDERING=... (ROWS=fixture.json | DATASET_CACHE=.data/huggingface) LEDGER=... PREREGISTRATION=... OUTPUT=... PROVIDER_MODEL=... BASE_URL=https://... TIMEOUT_SECONDS=... ADAPTER_REVISION=... PACKAGE_REVISION=typesafe-sdk-0.7.1 ATTEMPT_CEILING=... MAX_NEW=... CONFIRM=--confirm"; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.ordering_cli run --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" --initial-preflight "$(PREFLIGHT)" --initial-observations "$(INITIAL_OBSERVATIONS)" --ordering "$(ORDERING)" $(SOURCE_ARGS) --ledger "$(LEDGER)" --preregistration "$(PREREGISTRATION)" --output "$(OUTPUT)" --provider-model "$(PROVIDER_MODEL)" --base-url "$(BASE_URL)" --timeout-seconds "$(TIMEOUT_SECONDS)" --adapter-revision "$(ADAPTER_REVISION)" --package-revision "$(PACKAGE_REVISION)" --attempt-ceiling "$(ATTEMPT_CEILING)" --max-new "$(MAX_NEW)" --max-retries-per-request "$(MAX_RETRIES_PER_REQUEST)" $(RETRY_FAILED) $(RECOVER_UNCERTAIN) $(CONFIRM) $(OVERWRITE)

ordering-report:
	@if test -z "$(PROTOCOL)" || test -z "$(MANIFEST)" || test -z "$(ORDERING)" || test -z "$(OBSERVATIONS)" || test -z "$(OUTPUT)"; then \
		echo "Usage: make ordering-report PROTOCOL=... MANIFEST=... ORDERING=... OBSERVATIONS=... OUTPUT=... [BOOTSTRAP_SEED=0 RESAMPLES=1000 OVERWRITE=--overwrite]"; exit 2; \
	fi
	$(PYTHON) -m decision_flywheel_evaluations.ordering_cli report --protocol "$(PROTOCOL)" --manifest "$(MANIFEST)" --ordering "$(ORDERING)" --observations "$(OBSERVATIONS)" --output "$(OUTPUT)" $(if $(BOOTSTRAP_SEED),--seed "$(BOOTSTRAP_SEED)") $(if $(RESAMPLES),--resamples "$(RESAMPLES)") $(OVERWRITE)

install-tools:
	$(PYTHON) -m pip install -e '.[tools]'

release:
	$(PYTHON) -m semantic_release version

# Unified flywheel harness (studies/UNIFIED_FLYWHEEL_PLAN.md). Offline only: a fake Jev and fixed
# analyst replies, with every non-local socket blocked. UF_PYTHON must have scikit-learn and Tactus
# (Jev-Flywheel's interpreter does); the pinned Jev-Flywheel clone lives in var/ (see prepare-clone).
# UF_CORE is the Decision-Flywheel source tree the harness imports (default: the sibling working
# tree); its commit and cleanliness are recorded in every run summary. UF_CORE= uses the pinned copy.
UF_CORE ?= $(abspath ../Decision-Flywheel/src)

unified-flywheel-test:
	@if test -z "$(UF_PYTHON)"; then echo "Usage: make unified-flywheel-test UF_PYTHON=/path/to/python-with-scikit-learn-and-tactus"; exit 2; fi
	UNIFIED_FLYWHEEL_CORE=$(UF_CORE) $(UF_PYTHON) scripts/unified_flywheel.py check-env
	UNIFIED_FLYWHEEL_CORE=$(UF_CORE) PYTHONPATH=src:$(if $(UF_CORE),$(UF_CORE),var/unified-flywheel/pyshim) $(UF_PYTHON) -m pytest -q -p no:cacheprovider src/decision_flywheel_evaluations/unified_*_test.py

unified-flywheel-dry-run:
	@if test -z "$(UF_PYTHON)"; then echo "Usage: make unified-flywheel-dry-run UF_PYTHON=/path/to/python-with-scikit-learn-and-tactus"; exit 2; fi
	UNIFIED_FLYWHEEL_CORE=$(UF_CORE) $(UF_PYTHON) scripts/unified_flywheel.py run --run-dir var/unified-flywheel/dry-run $(UF_ARGS)

# These commands cannot collect live results or read the private SME policy.
reviews-stream-fixture:
	@if test -z "$(STREAM_FIXTURE)" || test -z "$(STREAM_RUN_DIR)" || test -z "$(STREAM_OUTPUT)" || test -z "$(STREAM_MAX_NEW)"; then \
		echo "Usage: make reviews-stream-fixture STREAM_FIXTURE=synthetic.json STREAM_RUN_DIR=var/fixture STREAM_OUTPUT=var/fixture.run.json STREAM_MAX_NEW=... [OVERWRITE=--overwrite]"; exit 2; \
	fi
	$(PYTHON) scripts/reviews_stream.py run --fixture "$(STREAM_FIXTURE)" --run-dir "$(STREAM_RUN_DIR)" --output "$(STREAM_OUTPUT)" --max-new-requests "$(STREAM_MAX_NEW)" $(OVERWRITE)

reviews-stream-report:
	@if test -z "$(STREAM_INPUT)" || test -z "$(STREAM_OUTPUT)"; then \
		echo "Usage: make reviews-stream-report STREAM_INPUT=var/fixture.run.json STREAM_OUTPUT=var/fixture.report.json [OVERWRITE=--overwrite]"; exit 2; \
	fi
	$(PYTHON) scripts/reviews_stream.py report --input "$(STREAM_INPUT)" --output "$(STREAM_OUTPUT)" $(OVERWRITE)
