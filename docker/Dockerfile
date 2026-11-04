# HydraY self-play datagen, with optional upload of the shards to Hugging Face.
#
#   docker build -f docker/Dockerfile -t hydray-datagen .
#   docker run --rm --init -v "$PWD/data:/data" hydray-datagen --nodes 8000
#
# ARCH picks the ISA as in the makefile; the default runs on any AVX2 + BMI2
# CPU. -march=native would tie the image to the build machine.

FROM debian:trixie-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends clang g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
ARG ARCH="-march=x86-64-v3"
ARG JOBS=2
# Static, like `make release`: the binary needs nothing from the final image.
RUN make prod -j"$JOBS" ARCH="$ARCH" RELEASE_FLAGS="-DHYDRAY_RELEASE -static -Wl,-u,log"

FROM python:3.13-slim
RUN pip install --no-cache-dir --root-user-action=ignore huggingface_hub==2.3.0
# Any --user works: the Hub client keeps its cache out of the missing $HOME.
ENV HF_HOME=/tmp/hf PYTHONUNBUFFERED=1
COPY --from=build /src/chess /hydray/chess
COPY nnue/tools/hf_upload.py /hydray/hf_upload.py
COPY docker/datagen.sh /hydray/datagen.sh
# datagen looks for tablebases at engine/syzygy/files relative to the cwd.
WORKDIR /hydray
VOLUME ["/data"]
ENTRYPOINT ["/hydray/datagen.sh"]
