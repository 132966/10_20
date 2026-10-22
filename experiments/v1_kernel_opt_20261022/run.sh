#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/sglang_sparse/activate_cuda_compat.sh
export PYTHONPATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/sglang_sparse/sglangenv/lib/python3.12/site-packages
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=2
/usr/local/bin/python3.12 test_build.py
/usr/local/bin/python3.12 make_report.py
