FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.cfg requirements.txt ./
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "-m", "src.main"]
