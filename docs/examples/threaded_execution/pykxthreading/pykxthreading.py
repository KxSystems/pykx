import threading
from queue import Queue


def q_thread(q_queue):
    import os
    os.environ['PYKX_NO_SIGINT'] = '1'
    import pykx as kx
    e_q = kx.EmbeddedQ()
    while True:
        call = q_queue.get()
        if callable(call):
            break
        fut = call[0]
        try:
            res = e_q(call[1], *call[2], **call[3])
            fut.set_result(res)
        except BaseException as e:
            fut.set_exception(e)


q_queue = Queue()
th = threading.Thread(target=q_thread, args=(q_queue,))
th.start()


import pykx as kx


class ThreadedQ(kx.Q):
    def __init__(self, q_queue):
        object.__setattr__(self, 'q_queue', q_queue)
        super().__init__()

    def __call__(self, query, *args, **kwargs):
        fut = kx.EmbeddedQFuture()
        self.q_queue.put((fut, query, args, kwargs))
        return fut._await()

    _call = __call__


q = ThreadedQ(q_queue)


def close():
    global th
    global q_queue
    q_queue.put(lambda x: 1)
    th.join()
