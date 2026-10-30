#!/bin/bash
# Start CARLA 0.9.16 server (Apptainer) headless on an Explorer GPU node.
# Usage: ./start_carla.sh [rpc_port] [extra CARLA args...]
#
# Fix for the UE4 crash ("GameThread timed out waiting for RenderThread", GenericGPUBrand):
# --nv injects the host's (RHEL 9) glvnd libEGL/libGLdispatch, which need GLIBC 2.33+,
# but the container is Ubuntu 20.04 (glibc 2.31). libGLX_nvidia then fails to init,
# Vulkan falls back to lavapipe (CPU) and the render thread times out.
#   1) Put the container's own glvnd ahead of /.singularity.d/libs
#   2) Point glvnd at NVIDIA's EGL vendor instead of Mesa

PORT=${1:-2000}
shift || true
CARLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Per-user runtime dir (must be 0700 and owned by the user), so anyone with
# access to this folder can start CARLA
XDG_DIR="/tmp/xdg-runtime-$USER"
mkdir -p "$XDG_DIR" && chmod 700 "$XDG_DIR"

exec apptainer exec --nv \
  --env XDG_RUNTIME_DIR="$XDG_DIR" \
  --bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json:/etc/vulkan/icd.d/nvidia_icd.json \
  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/opt/10_nvidia.json \
  --env __EGL_VENDOR_LIBRARY_FILENAMES=/opt/10_nvidia.json \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs \
  "$CARLA_DIR/carla-0.9.16.sif" \
  /workspace/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 \
  -RenderOffScreen -nosound -carla-rpc-port="$PORT" "$@"
