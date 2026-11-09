PY ?= python3
PORT ?= 8000

.PHONY: help install setup index test eval eval-retrieval eval-answers report serve demo doctor clean

help:
	@echo "make install         install Python dependencies"
	@echo "make setup           clone the 3 evaluation repos (pinned versions)"
	@echo "make index           build AST + fixed indexes with code + general embeddings"
	@echo "make test            run the test suite (offline, no API keys needed)"
	@echo "make eval            retrieval eval + answer eval + report"
	@echo "make serve           start the web app on http://127.0.0.1:$(PORT)"
	@echo "make demo            everything above, then open the browser"

install:
	$(PY) -m pip install -r requirements.txt

setup:
	$(PY) -m cqa setup

index: setup
	$(PY) -m cqa index --all

test:
	$(PY) -m pytest

eval-retrieval:
	$(PY) -m cqa eval retrieval

eval-answers:
	$(PY) -m cqa eval answers

report:
	$(PY) -m cqa eval report

eval: eval-retrieval eval-answers

serve:
	$(PY) -m cqa serve --port $(PORT)

doctor:
	$(PY) -m cqa doctor

demo:
	./run_demo.sh

clean:
	rm -rf data/indexes data/cache/llm .pytest_cache
