FROM python:3.7

WORKDIR /app

COPY . .
RUN pip3 install -r requirements.txt

EXPOSE 8081

CMD ["uvicorn", "turkishdelightnlp.main:app",  "--host", "0.0.0.0", "--port", "8000"]