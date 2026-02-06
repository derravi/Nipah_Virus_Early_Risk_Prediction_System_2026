FROM python:3.13.7

RUN mkdir -p nipah_virus_prediction_model

WORKDIR /nipah_virus_prediction_model

COPY . /nipah_virus_prediction_model

RUN pip install -r requirements.txt

ENV PORT=800

EXPOSE 8000

CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8000"]