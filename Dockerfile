FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml .
RUN uv sync --no-cache --no-dev

COPY . .

EXPOSE 5000

CMD ["uv", "run", "--no-sync", "python", "src/main.py"]
