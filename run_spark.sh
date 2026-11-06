#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-$(command -v python)}"
SPARK_VER=3.5.9
exec spark-submit --master 'local[3]' --driver-memory 2g \
  --packages "org.apache.spark:spark-sql-kafka-0-10_2.12:${SPARK_VER}" \
  spark_kafka_stream.py "$@"
