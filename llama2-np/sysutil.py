import psutil


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
