FROM python:3.11-slim
WORKDIR /app
COPY deployment/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY deployment/app.py app.py
COPY model_building model_building
COPY data/processed data/processed
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
