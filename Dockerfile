FROM loris3/cuda:latest

WORKDIR /srv/home/users/kalinchukd23cs/InfEstimation_benchmark

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    traker==0.3.2 \
    accelerate==1.13.0 \
    numpy==2.0.2 \
    pandas==2.2.2 \
    datasets==4.0.0 \
    peft==0.18.1 \
    transformers==5.0.0 \
    torch==2.10.0+cpu \
    torchvision==0.25.0+cpu \
    tqdm==4.67.3 \
    --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["python3"]

LABEL org.opencontainers.image.source="https://github.com/DarynaKalinchuk/InfEstimation_benchmark"
