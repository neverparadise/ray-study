import time
import ray
ray.init()

database = [
    "Learning", "Ray", "Flexible", "Distributed", "Python", 
    "for", "Data", "Science"
]

# ex 2-1
def retrieve(item):
    #time.sleep(item / 10.)
    time.sleep(len(database[item]) / 10.0)
    return item, database[item]

def print_runtime(input_data, start_time, decimals=2):
    print(f"Runtime: {time.time() - start_time:.{decimals}f} seconds, data:")
    print(*input_data, sep="\n")

start = time.time()
data = [retrieve(item) for item in range(8)]
print_runtime(data, start)

# ex 2-2: ray remote decorator
@ray.remote # 데코레이터를 사용하면 파이썬 함수를 Ray task로 만들 수 있다. 
def retrieve_task(item):
    return retrieve(item)

# 처음에 파이썬 코드를 작성하고 제대로 동작시키는 것에 집중한다.
# 데코레이터를 코드에 추가한다. (wrapping)
# 

start = time.time()
data_references = [retrieve_task.remote(item) for item in range(8)]
data = ray.get(data_references)
print_runtime(data, start, 2)

# ex 2-3: Object store with put and get
# 현재 상황은 retrieve 함수가 database에 직접 접근할 수 있다. 그래서 remote function 또한 local Ray cluster에 접근하면서 동작한다.
# 하지만 만약 여러개의 컴퓨터로 구성된 actual cluster 환경에서는 이렇게 직접 접근해서 동작하지 못한다. 
# 다행히 ray에서는 driver나 여러 개의 worker 사이에서 데이터를 공유하는 쉬운 방법을 제공한다.
# ray의 put함수를 통해 Head node나 Worker node의 Objectstore에 데이터를 넣고 레퍼런스 주소 값을 받아올 수 있다.
# ray의 get함수를 통해 레퍼런스 주소값을 넣어서 원래 데이터를 가져올 수 있다. 
# 이렇게 오브젝트 스토어에 데이터를 공유하면 오버헤드가 줄어든다.

database_object_ref = ray.put(database)

@ray.remote
def retrieve_task(item):
    obj_store_data = ray.get(database_object_ref)
    time.sleep(len(obj_store_data[item]) / 10.0)
    return item, obj_store_data[item]

start = time.time()
data_references = [retrieve_task.remote(item) for item in range(8)]
data = ray.get(data_references)
print_runtime(data, start, 2)




# Wait function for non-blocking calls
# 예제 2-2처럼 ray.get(data_references)를 사용해서 결과에 접근하는 것을 blocking 이라고 부른다. 
# 이는 Head node의 driver process가 모든 결과들이 끝날 때 까지 기다린다는 의미이다. 
# 지금의 문제에서는 큰 문제가 되지 않지만 각 데이터를 처리하는 시간이 몇 분씩 걸릴 수도 있다. 
# 이 때, 우리는 driver process가 대기하기보다는 다른 task들에 대해서 자유롭게 동작하기를 원한다. 
# 그리고 드라이버가 작업이나 데이터가 들어올 때마다 처리하면 정말 좋을 것이다.  
# 또 생각해봐야할 것은 데이터를 탐색하지 못한다면 무슨일이 생길까? 데드락이 걸린다. 
# 데이터베이스 연결에 데드락이 발생하면 드라이버는 멈추게 되고 모든 아이템을 탐색할 수 없게 된다. 
# 따라서 적절한 타임아웃으로 동작시키는 것이 좋은 생각이다. 
# 

# ex 2-4: ray.wait 

start = time.time()
data_references = [retrieve_task.remote(item) for item in range(8)]
all_data = []

while len(data_references) > 0:
    finished, data_references = ray.wait(data_references, num_returns=2, timeout=7.0)
    data = ray.get(finished)
    print_runtime(data, start, 3)
    all_data.extend(data)

# Handling task dependencies
# 이제 데이터가 로드되고 나서 수행되는 follow up task를 실행하는 상황을 가정하자. 

@ray.remote
def follow_up_task(retrieve_result): # retrieve 함수의 리턴 값 2개를 받는다. 
    original_item, _ = retrieve_result # 튜플을 언패킹한다. 
    follow_up_result = retrieve(original_item + 1) 
    return retrieve_result, follow_up_result

retrieve_refs = [retrieve_task.remote(item) for item in [0, 2, 4, 6]]
follow_up_refs = [follow_up_task.remote(ref) for ref in retrieve_refs] 
# 여기서 주목할 것은 함수의 인풋으로 튜플이 아니라 레퍼런스를 넣은 것이다. 
# retrieve_refs에 있는 오브젝트 레퍼런스를 전달한다. 
# Ray는 follow_up_task가 실제 값을 필요로 한다는 것을 알고 있다. 
# 그래서 이 테스크에서 내부적으로 ray.get을 호출해서 Future를 해결한다
# future객체는 지연된 계산을 표현하기 위해 사용된다. 그 객체의 계산은 완료되었을 수도 있고 되지 않았을 수도 있다. 
# ray는 내부적으로 모든 작업들에 대한 dependency graph를 만든다. 그리고 의존성과 관련된 순서로 모든 것을 실행한다. 

result = [print(data) for data in ray.get(follow_up_refs)]


# From classes to actors
# 클래스 형태로 Actor를 구현하면 클러스터에서 현재 상태를 표현할 수 있는 처리를 구현할 수 있다. 

# ex 2-5: 

@ray.remote
class DataTracker:
    def __init__(self) -> None:
        self._counts = 0
    
    def increment(self):
        self._counts += 1
    
    def counts(self):
        return self._counts

# 이 클래스는 데이터 개수를 추적한다. 

@ray.remote
def retrieve_tracker_task(item, tracker):
    obj_store_data = ray.get(database_object_ref)
    time.sleep(item / 10.)
    tracker.increment.remote()
    return item, obj_store_data[item]

tracker = DataTracker.remote()
data_references = [retrieve_tracker_task.remote(item, tracker) for item in range(8)]
print(ray.get(data_references))
print(ray.get(tracker.counts.remote()))
print(ray.get(tracker._counts))

# Ray tasks and actors decorated by @ray.remote are running in different processes that don’t share the same address space as ray driver (Python script that runs ray.init).
# That says if you define a global variable and change the value inside a driver, changes are not reflected in the workers


# Ownership
@ray.remote
def task_owned():
    return

@ray.remote
def task(dependency):
    res_owned = task_owned.remote()
    return

val = ray.put("Value")
res = task.remote(dependency=val)
