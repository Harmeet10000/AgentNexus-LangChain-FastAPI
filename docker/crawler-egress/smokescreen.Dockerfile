FROM alpine:3.22.2@sha256:4b7ce07002c69e8f3d704a9c5d6fd3053be500b7f1c69fc0d80990c2ad8dd412

ARG TARGETARCH
ARG SMOKESCREEN_VERSION=v0.1.0
ARG SMOKESCREEN_SHA256_AMD64=4e1e447449e3cc653dae956208588722b5c8fe8ce894f633adc756a851d1ec46
ARG SMOKESCREEN_SHA256_ARM64=f15ddd0dc2b2cbc6d7318f441928939eb9bedfab2a1881508af3096a1e24b7c9

RUN apk add --no-cache ca-certificates \
    && case "${TARGETARCH}" in \
         amd64) archive_sha256="${SMOKESCREEN_SHA256_AMD64}" ;; \
         arm64) archive_sha256="${SMOKESCREEN_SHA256_ARM64}" ;; \
         *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && archive="smokescreen_${SMOKESCREEN_VERSION}_linux_${TARGETARCH}.tar.gz" \
    && wget -q -O "/tmp/${archive}" \
         "https://github.com/stripe/smokescreen/releases/download/${SMOKESCREEN_VERSION}/${archive}" \
    && echo "${archive_sha256}  /tmp/${archive}" | sha256sum -c - \
    && mkdir -p /tmp/smokescreen \
    && tar -xzf "/tmp/${archive}" -C /tmp/smokescreen \
    && install -m 0755 "$(find /tmp/smokescreen -type f -name smokescreen -print -quit)" \
         /usr/local/bin/smokescreen \
    && addgroup -S smokescreen \
    && adduser -S -D -H -G smokescreen -u 65532 smokescreen \
    && rm -rf /tmp/smokescreen "/tmp/${archive}"

USER 65532:65532
EXPOSE 4750 9810
ENTRYPOINT ["/usr/local/bin/smokescreen"]
