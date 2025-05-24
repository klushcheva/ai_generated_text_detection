import os
from redis import Redis
from rq import Worker, Queue

# Configuration
listen = ['text_classification_queue']
redis_conn = Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=6379,
    db=0,
    decode_responses=False
)

if __name__ == '__main__':
    # Create queues
    queues = [Queue(name, connection=redis_conn) for name in listen]

    # Start worker
    worker = Worker(queues, connection=redis_conn)
    worker.work()