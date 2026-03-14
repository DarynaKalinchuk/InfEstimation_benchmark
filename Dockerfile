FROM huggingface/transformers-pytorch-gpu:latest

WORKDIR /srv/home/users/kalinchukd23cs/InfEstimation_benchmark

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    traker==0.3.2 \
    peft==0.18.1
    
CMD ["python3"]

LABEL org.opencontainers.image.source="https://github.com/DarynaKalinchuk/InfEstimation_benchmark"
