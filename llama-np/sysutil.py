import psutil
import logging

from config import ExperimentArgs

'''
The meaning of values in the tuple returned by psutil.virtual_memory
0 - total: total memory excluding swap
1 - available: available memory for processes
2 - percent: memory usage in percent
3 - used: the memory used
4 - free: memory not used at and is readily available
'''

def report_mem(exp_args: ExperimentArgs):
    print("== CPU Memory Report ==")
    print('RAM total:', psutil.virtual_memory()[0]/1000000000)
    print('RAM used (GB):', psutil.virtual_memory()[3]/1000000000)
    print('RAM % used:', psutil.virtual_memory()[2])
    print('RAM available (GB):', psutil.virtual_memory()[1]/1000000000)
    print('RAM free (GB):', psutil.virtual_memory()[4]/1000000000)

    if exp_args.use_cupy:
        import cupy
        mempool = cupy.get_default_memory_pool()
        print("== GPU Memory Report ==")
        print("Default pool - limit (MB) :", mempool.get_limit()/1024/1024)              
        print("Default pool - total (MB) :", mempool.total_bytes()/1024/1024)              
        print("Default pool - used (MB)  :", mempool.used_bytes()/1024/1024)              
        print("Default pool - free (MB)  :", mempool.free_bytes()/1024/1024)              
        print("Default pool - free blocks:", mempool.n_free_blocks())              
        mempool = cupy.get_default_pinned_memory_pool()
        print(" Pinned pool - free blocks:", mempool.n_free_blocks())              


# When using python logger.debug()/info()/warning()/error()/critical() functions, the arguments will be evaluated
# before the function is called, regardless of the log level. This side effect can be problematic, for example:
# 
#      z = cupy.matmul(x, y)
#      logger.debug(f"check {z[1][1]}")
# will always result in a D2H copy of z, even if the log level is not DEBUG.
# 
# The lumi_logging class here avoids this problem by using callable as the argument. 
#
# Instead of 
#   logger.debug(f"check {z[1][1]}") 
# Use
#   lumi_logger.debug(lambda: f"check {z[1][1]}") 

class lumi_logger:
    def __init__(self, *args, **kargs):
        self.logger = logging.getLogger(*args, **kargs) 

    def debug(self, callable, *args, **kargs):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(callable(), *args, **kargs)
        return

    def info(self, callable, *args, **kargs):
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(callable(), *args, **kargs)
        return

    def error(self, callable, *args, **kargs):
        if self.logger.isEnabledFor(logging.ERROR):
            self.logger.error(callable(), *args, **kargs)
        return

    def warning(self, callable, *args, **kargs):
        if self.logger.isEnabledFor(logging.WARNING):
            self.logger.warning(callable(), *args, **kargs)
        return

    def critical(self, callable, *args, **kargs):
        if self.logger.isEnabledFor(logging.CRITICAL):
            self.logger.critical(callable(), *args, **kargs)
        return

class lumi_logging:
    @classmethod
    def basicConfig(cls, *args, **kargs):
        logging.basicConfig(*args, **kargs)

    @classmethod
    def getLogger(cls, *args, **kargs):
        return lumi_logger(*args, **kargs)

    
