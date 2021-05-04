FROM doduo1.umcn.nl/uokbaseimage/diag
RUN pip3 install torchsummary
RUN pip3 install wandb --upgrade
RUN pip3 install pytorch-lightning
ENV CODEBASE=/mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560
WORKDIR /mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560/msk-compression-fracture-detection
#ENTRYPOINT python3.8 main.py --test