import platform
import os
import psutil

def get_system_info():
    system_info = platform.uname()
    
    # Basic information from platform module
    print("System: {}".format(system_info.system))
    print("Node Name: {}".format(system_info.node))
    print("Release: {}".format(system_info.release))
    print("Version: {}".format(system_info.version))
    print("Machine: {}".format(system_info.machine))
    print("Processor: {}".format(system_info.processor))

    # More detailed information using os and psutil
    print("\nCPU Cores: {}".format(psutil.cpu_count(logical=False)))
    print("Logical CPUs: {}".format(psutil.cpu_count(logical=True)))
    
    cpu_frequency = psutil.cpu_freq()
    print("CPU Frequency: {} MHz".format(cpu_frequency.current))

    cpu_usage_percent = psutil.cpu_percent(interval=1, percpu=True)
    print("\nCPU Usage:")
    for i, percent in enumerate(cpu_usage_percent):
        print("Core {}: {}%".format(i, percent))

    # Memory information
    memory_info = psutil.virtual_memory()
    print("\nMemory:")
    print("Total: {:.2f} GB".format(memory_info.total / (1024 ** 3)))
    print("Available: {:.2f} GB".format(memory_info.available / (1024 ** 3)))
    print("Used: {:.2f} GB".format(memory_info.used / (1024 ** 3)))
    print("Percentage Used: {}%".format(memory_info.percent))

if __name__ == "__main__":
    get_system_info()

