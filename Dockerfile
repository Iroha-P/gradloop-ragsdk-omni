FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BAOYAN_PROFILE=local_baseline \
    BAOYAN_RAG_BACKEND=baseline \
    BAOYAN_LLM_BACKEND=none \
    BAOYAN_PUBLIC_DEMO=true

WORKDIR /app
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/runtime \
    && chown app:app /app/runtime
COPY pyproject.toml README.md ./
COPY app ./app
COPY data/public_eval/portfolio_v1/corpus.jsonl ./data/public_eval/portfolio_v1/corpus.jsonl
COPY data/public_demo/question_bank.jsonl ./data/public_demo/question_bank.jsonl
RUN pip install --no-cache-dir .
COPY --chown=app:app scripts/run_local.py scripts/run_public_demo.py ./scripts/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
