FROM doduo1.umcn.nl/uokbaseimage/diag
RUN pip3 install torchsummary
RUN pip3 install wandb --upgrade
ENV CODEBASE=/mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560
