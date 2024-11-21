import random
import heapq


class PreemptiveSemaphore:
    def __init__(self, initial):
        self.value = initial
        self.waiting = []  # Min-heap for waiting threads, ordered by priority

    def acquire(self, thread_id, priority):
        if self.value > 0:
            self.value -= 1
            print(f"Thread {thread_id} acquired the semaphore.")
        else:
            print(
                f"Thread {thread_id} with priority {priority} is waiting for the semaphore."
            )
            heapq.heappush(self.waiting, (priority, thread_id))

    def release(self, thread_id):
        if self.waiting:
            # Pop the highest priority thread (smallest priority number)
            next_priority, next_thread = heapq.heappop(self.waiting)
            print(
                f"Thread {thread_id} released the semaphore; now Thread {next_thread} with priority {next_priority} can acquire it."
            )
            self.value -= 1  # Immediately allocate the semaphore to the next thread
        else:
            self.value += 1
            print(f"Thread {thread_id} released the semaphore.")


def simulate_thread(semaphore, thread_id, priority):
    # Try to acquire the semaphore
    semaphore.acquire(thread_id, priority)
    # Simulate work completion (trigger release immediately)
    semaphore.release(thread_id)


def main():
    semaphore = PreemptiveSemaphore(3)  # Allow 3 concurrent accesses

    # Simulate 10 threads with random priorities
    threads = []
    for i in range(10):
        priority = random.randint(1, 10)  # Random priority between 1 and 10
        threads.append((i, priority))

    # Sort threads by priority before processing
    threads.sort(key=lambda x: x[1])  # Sort by priority for simulation

    for thread_id, priority in threads:
        simulate_thread(semaphore, thread_id, priority)


if __name__ == "__main__":
    main()
