# News sentiment classifier 

This image consists of a lightweight streamlit dashboard hooked up with a custom sentiment classifier capable of fetching news and displaying the results from NewsAPI.

### Compose File

```yaml
version: '3.8'
services:
  app:
    build: .
    container_name: news-sentiment-classifier-app
    # Map Streamlit default port 8501 for the dashboard
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
      - MODEL_DIR=/app/news_sentiment_model
    command: ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

    develop:
      watch:
        - action: sync
          path: .
          target: /app

```

Run using `docker compose up --watch`