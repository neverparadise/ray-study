import ray
import time
ray.init()

items = [{"name": str(i), "data": i} for i in range(10000)]
ds = ray.data.from_items(items)
ds.show(5)

# ex1-2
start = time.time()
squares = ds.map(lambda x: x["data"] ** 2)
evens = squares.filter(lambda x: x % 2 == 0)
evens.count()
cubes = evens.flat_map(lambda x: [x, x**3])
sample = cubes.take(10)
print(sample)
end = time.time()
print(end - start)


# ex1-3
start = time.time()
pipe = ds.window()
result = pipe.map(lambda x: x["data"] ** 2).filter(lambda x: x % 2 == 0).flat_map(lambda x: [x, x**3])
result.show()
end = time.time()
print(end - start)

ray.shutdown()
