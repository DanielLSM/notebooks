from multiprocessing import Process, Lock
import time

def f(l, i):
    l.acquire()
    try:
    	if not i:
    		time.sleep(2)
    	print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()