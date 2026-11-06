import argparse
import json
import time

from config import PROCESSED_DATA
from features import load_processed


def event_generator(path=PROCESSED_DATA, delay=0.0, limit=None):
    """Yield supplied ISS rows one at a time as stream events.

    `delay` controls replay speed. Use 10.0 to mimic the original sampling
    interval, or a smaller value for a short classroom demonstration.
    """
    df = load_processed(path)
    if limit:
        df = df.iloc[:limit]
    for _, row in df.iterrows():
        event = row.to_dict()
        event["timestamp"] = str(event["timestamp"])
        yield event
        if delay:
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Replay the supplied ISS dataset as JSON stream events.")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between emitted events")
    parser.add_argument("--limit", type=int, default=20, help="Number of events to print; 0 means all")
    args = parser.parse_args()
    limit = None if args.limit == 0 else args.limit
    for event in event_generator(delay=args.delay, limit=limit):
        print(json.dumps(event, default=str))


if __name__ == "__main__":
    main()
