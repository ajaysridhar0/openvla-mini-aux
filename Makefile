.PHONY: test manifest-check lock-check

test:
	uv run --locked python -m unittest discover -s tests -v

manifest-check:
	uv run --locked python scripts/build_data_manifest.py $(DATA_ROOT) --output /tmp/barx-manifest.csv

lock-check:
	uv lock --check
