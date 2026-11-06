"""Create / inspect / reset the REAL three-partition Kafka ISS topic."""
import argparse
import os
from kafka import KafkaAdminClient, KafkaConsumer
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError

TOPIC = os.getenv("ISS_KAFKA_TOPIC", "iss_telemetry")
BOOTSTRAP = os.getenv("ISS_KAFKA_BOOTSTRAP", "localhost:9092")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete the old topic and create an empty one")
    args = parser.parse_args()
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, client_id="iss-setup")
    if args.reset and TOPIC in admin.list_topics():
        admin.delete_topics([TOPIC])
        print("Deleted", TOPIC, "(topic removal is asynchronous; run setup again if needed)")
        admin.close()
        return
    if TOPIC not in admin.list_topics():
        try:
            admin.create_topics([NewTopic(name=TOPIC, num_partitions=3, replication_factor=1)])
        except TopicAlreadyExistsError:
            pass
    admin.close()
    consumer = KafkaConsumer(bootstrap_servers=BOOTSTRAP, consumer_timeout_ms=1000)
    partitions = consumer.partitions_for_topic(TOPIC)
    print(f"Kafka topic {TOPIC!r}: partitions={sorted(partitions or [])}, broker={BOOTSTRAP}")
    consumer.close()
    if not partitions or len(partitions) != 3:
        raise RuntimeError("Expected exactly 3 partitions; check the existing topic configuration")


if __name__ == "__main__":
    main()
