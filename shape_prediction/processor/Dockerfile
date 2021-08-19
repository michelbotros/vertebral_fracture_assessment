FROM doduo1.umcn.nl/uokbaseimage/diag

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm
COPY --chown=algorithm:algorithm src/ /opt/algorithm/src/
COPY --chown=algorithm:algorithm checkpoints/ /opt/algorithm/checkpoints/
WORKDIR /opt/algorithm/src/

ENTRYPOINT python3.8 pipeline.py