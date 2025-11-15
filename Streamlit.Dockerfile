FROM python:3.12.7-slim
WORKDIR /app

RUN mkdir ./FinAgent
COPY pyproject.toml /app/
COPY FinAgent/__init__.py /app/FinAgent/
COPY README.md /app/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
COPY FinAgent/ /app/FinAgent/
COPY ["app.py", "agent_page.py", "explainability_agent_page.py", "logging.yaml", "API Key Instructions.md", "/app/"]
COPY Dockerfile /app/
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--browser.serverAddress", "0.0.0.0", "--server.port", "8501"]