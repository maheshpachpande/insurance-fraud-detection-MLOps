# ---- builder ----
FROM python:3.10 as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# ---- runtime ----
FROM python:3.10-alpine

WORKDIR /app
COPY --from=builder /install /home/mahesh/Desktop/insurance-fraud-detection-MLOps/venv1/lib/python3.10/site-packages
COPY . .
ENTRYPOINT ["python", "app.py"]
