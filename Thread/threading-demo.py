import threading
import time

def thread_job():
    print("This is an added Thread,number is %s" % threading.current_thread())
    print('T1 start\n')
    for i in range(10):
        time.sleep(0.3)
    print("T1 finish\n")

def main():
    added_thread = threading.Thread(target=thread_job,name='T1')
    added_thread.start()
    # print(threading.active_count)
    # print(threading.enumerate())
    # print(threading.current_thread())
    added_thread.join()  # join的用法是将这个线程join到主线程，然后就要等这个子线程先跑完，再跑主线程。
    print("all done\n")

if __name__=='__main__':
    main()