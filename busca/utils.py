import os
import sys

try:
    import psutil
except ImportError:
    import resource


def get_ram_usage():
    if 'psutil' in sys.modules:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    else:
        process = resource.getrusage(resource.RUSAGE_SELF)
        return process.ru_maxrss * 1024

def get_total_ram():
    if 'psutil' in sys.modules:
        return psutil.virtual_memory().total
    else:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')





