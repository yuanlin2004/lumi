import psutil
import logging


'''
The meaning of values in the tuple returned by psutil.virtual_memory
0 - total: total memory excluding swap
1 - available: available memory for processes
2 - percent: memory usage in percent
3 - used: the memory used
4 - free: memory not used at and is readily available
'''

def report_mem():
    # Importing the library
    print('RAM total:', psutil.virtual_memory()[0]/1000000000)
    print('RAM used (GB):', psutil.virtual_memory()[3]/1000000000)
    print('RAM % used:', psutil.virtual_memory()[2])
    print('RAM available (GB):', psutil.virtual_memory()[1]/1000000000)
    print('RAM free (GB):', psutil.virtual_memory()[4]/1000000000)


# When using python logger.debug()/info()/warning()/error()/critical() functions, the arguments will be evaluated
# before the function is called, regardless of the log level. This side effect can be problematic, for example:
# 
#      z = cupy.matmul(x, y)
#      logger.debug(f"check {z[1][1]}")
# will always result in a D2H copy of z, even if the log level is not DEBUG.
# 
# The lumi_logging class here avoids this problem by using callable as the argument. 

# Instead of 
#   logger.debug(f"check {z[1][1]}") 
# Use
#   lumi_logger.debug(lambda: f"check {z[1][1]}") # test(1) is called only if the log level is DEBUG.

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

    
