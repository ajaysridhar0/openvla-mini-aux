.PHONY: test lock-check

test:
	uv run --locked python -m unittest discover -s tests -v

lock-check:
	uv lock --check
