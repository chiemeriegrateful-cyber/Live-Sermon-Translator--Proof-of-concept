import asyncio
import itertools
from collections import deque


class Broker:
    def __init__(self, ring_size=30):
        self._subscribers = []
        self._ring = deque(maxlen=ring_size)
        self._seq_counter = itertools.count(1)

    def subscribe(self):
        queue = asyncio.Queue(maxsize=5)
        self._subscribers.append(queue)
        return queue, list(self._ring)

    def unsubscribe(self, queue):
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def publish(self, message):
        message = {**message, "seq": next(self._seq_counter)}
        self._ring.append(message)

        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                queue.get_nowait()
                queue.put_nowait(message)

        return message
