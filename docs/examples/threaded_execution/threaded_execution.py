import threading

from pykxthreading import pykx as kx, shutdown_q


def main():
    a = [threading.Thread(target=lambda x: print(kx.q.til(x)), args=(x,)) for x in range(2, 13)]
    [x.start() for x in a]
    [x.join() for x in a]


if __name__ == '__main__':
    try:
        main()
    finally:
        # Must shutdown the background thread to properly exit
        shutdown_q()
