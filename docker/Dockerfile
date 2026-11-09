# HydraY self-play datagen, with optional upload of the shards to Hugging Face.
#
#   docker build -f docker/Dockerfile -t hydray-datagen .
#   docker run --rm --init -v "$PWD/data:/data" hydray-datagen --nodes 8000
#
# ARCH picks the ISA as in the makefile, FALLBACK_ARCH a second build that
# datagen.sh runs where the first one does not start (no AVX-VNNI); an empty
# FALLBACK_ARCH builds only the first. -march=native would tie the image to
# the build machine.

FROM debian:trixie-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends clang g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
ARG ARCH="-march=x86-64-v3 -mavxvnni"
ARG FALLBACK_ARCH="-march=x86-64-v3"
ARG JOBS=2
# Static, like `make release`: the binaries need nothing from the final image.
# Each goes to bin/ with its ISA beside it.
RUN mkdir bin \
    && for b in main fallback; do \
           [ $b = main ] && arch=$ARCH || arch=$FALLBACK_ARCH; \
           [ -n "$arch" ] || continue; \
           make prod -j"$JOBS" ARCH="$arch" RELEASE_FLAGS="-DHYDRAY_RELEASE -static -Wl,-u,log" \
               && mv chess bin/$b && echo "$arch" > bin/$b.arch || exit 1; \
       done
# Recorded in every shard manifest: the version, a hash of the sources (net
# included) and, when the build passes it, the git commit; datagen.sh adds
# the ISA of the binary it runs.
ARG GIT_COMMIT=
RUN echo "hydray-$(sed -n 's/.*id name HydraY \([0-9.]*\).*/\1/p' uci/uci.cpp)" \
         "src-$(find board engine nnue tt uci main.cpp makefile -type f | LC_ALL=C sort \
                | xargs sha256sum | sha256sum | cut -c1-12)" \
         "${GIT_COMMIT:+git-$GIT_COMMIT}" > BUILD_ID

FROM python:3.13-slim
RUN pip install --no-cache-dir --root-user-action=ignore huggingface_hub==2.3.0
# Any --user works: the Hub client keeps its cache out of the missing $HOME.
# No progress bars: in docker logs they are pages of carriage returns.
ENV HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
COPY --from=build /src/BUILD_ID /hydray/
COPY --from=build /src/bin/ /hydray/bin/
COPY nnue/tools/hf_upload.py /hydray/hf_upload.py
COPY docker/datagen.sh engine/syzygy/fetch_syzygy.py engine/syzygy/syzygy-3-4-5.sha256 /hydray/
# datagen looks for tablebases at engine/syzygy/files relative to the cwd.
WORKDIR /hydray
VOLUME ["/data"]
ENTRYPOINT ["/hydray/datagen.sh"]
