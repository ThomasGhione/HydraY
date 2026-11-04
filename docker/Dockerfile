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
# Recorded in every shard manifest: the version, a hash of the sources (net
# included) and, when the build passes it, the git commit.
ARG GIT_COMMIT=
RUN echo "hydray-$(sed -n 's/.*id name HydraY \([0-9.]*\).*/\1/p' uci/uci.cpp)" \
         "src-$(find board engine nnue tt uci main.cpp makefile -type f | LC_ALL=C sort \
                | xargs sha256sum | sha256sum | cut -c1-12)" \
         "${GIT_COMMIT:+git-$GIT_COMMIT} $ARCH" > BUILD_ID

FROM python:3.13-slim
RUN pip install --no-cache-dir --root-user-action=ignore huggingface_hub==2.3.0
# Any --user works: the Hub client keeps its cache out of the missing $HOME.
ENV HF_HOME=/tmp/hf PYTHONUNBUFFERED=1
COPY --from=build /src/chess /src/BUILD_ID /hydray/
COPY nnue/tools/hf_upload.py /hydray/hf_upload.py
COPY docker/datagen.sh /hydray/datagen.sh
# datagen looks for tablebases at engine/syzygy/files relative to the cwd.
WORKDIR /hydray
VOLUME ["/data"]
ENTRYPOINT ["/hydray/datagen.sh"]
