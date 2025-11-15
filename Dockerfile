FROM python:3.12.7-slim
WORKDIR /app

RUN pip install fastapi==0.115.4
RUN mkdir ./FinAgent
COPY pyproject.toml /app/
COPY FinAgent/__init__.py /app/FinAgent/
COPY README.md /app/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir 'uvicorn[standard]'
COPY FinAgent/ /app/FinAgent/
COPY ["main.py", "logging.yaml", "API Key Instructions.md", "/app/"]
COPY Dockerfile /app/
EXPOSE 1524
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1524"]