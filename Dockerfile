FROM pytorch/pytorch

COPY './requirements.txt' .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Setup container directories
RUN mkdir /app

# Copy local code to the container
COPY ./app /app

WORKDIR /app
EXPOSE 8080
# ENTRYPOINT ["python"]
# CMD ["app/main.py"]
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]