#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/sglang_sparse/activate_cuda_compat.sh
export PYTHONPATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/sglang_sparse/sglangenv/lib/python3.12/site-packages
/usr/local/bin/python3.12 -u test_published_end.py 2>&1 | tee published_end.log
/usr/local/bin/python3.12 -u test_bench.py 2>&1 | tee run.log
/usr/local/bin/python3.12 -u test_prepared.py 2>&1 | tee prepared.log
/usr/local/bin/python3.12 -u test_consumer.py 2>&1 | tee consumer.log
/usr/local/bin/python3.12 make_report.py
