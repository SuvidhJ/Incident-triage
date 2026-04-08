setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	pytest -q

serve:
	uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload

inference:
	python inference.py

docker:
	docker build -t incident-triage-orchestrator .

docker-run:
	docker run -p 7860:7860 incident-triage-orchestrator

precheck:
	pytest -q
	python inference.py
	docker build -t incident-triage-orchestrator .
