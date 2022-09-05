import ray
import time

@ray.remote
def myfunc(num):
    time.sleep(0.1)
    sq = num ** 2
    return sq

num = 0

def myfunc2(num):
    time.sleep(0.1)
    sq = num ** 2
    return sq

start = time.time()
sq_lst = []
for i in range(100):
    sq = myfunc2(i)
    sq_lst.append(sq)
end = time.time()
print(sq_lst)
print(end - start)

start = time.time()
refs = [myfunc.remote(i) for i in range(100)]
end = time.time()
print(ray.get(refs))
print(end - start)

