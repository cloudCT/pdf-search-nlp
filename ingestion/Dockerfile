FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Add this line to ensure environment variables are accessible
ENV HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}

ENTRYPOINT ["python", "fetch_arxiv.py"]
