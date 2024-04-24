import functools
import time
import inspect

leading = 0
enabled = False

def decotimer_set(v):
    global enabled
    enabled = v

'''
About dynamically enable/disable the decorator timer

When Python interprets a function definition that is preceded by a decorator, 
the decorator is applied immediately. This means that the code within the decorator 
function runs as soon as the function it decorates is defined. This process transforms 
the decorated function according to the logic of the decorator, before the decorated 
function is ever called. Therefore the decoration may be done before `decotimer_set()` 
is executed. 

So the following method does not work.

def decotimer(func):
    def wrapper_timer():
        ...
    if enabled:
        return wrapper_timer
    else:
        return func
'''

def decotimer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        global leading
        leading +=1
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        leading -=1

        # Check if the function is a method of a class
        if inspect.ismethod(func):
            # If it's a method, prepend the class name
            class_name = func.__self__.__class__.__name__
            print_name = f"{class_name}.{func.__name__}"
        elif inspect.isfunction(func) and args and inspect.isclass(args[0].__class__):
            # This handles the case for class and static methods when used as decorators
            # where the first argument would be either the class (class method)
            # or an instance of the class (instance method).
            class_name = args[0].__class__.__name__
            print_name = f"{class_name}.{func.__name__}"
        else:
            # Otherwise, it's just a regular function
            print_name = func.__name__
        
        text = " " * (2*leading) + f"{print_name}() : {run_time:.4f} seconds"
        print(text)
        return value
    return wrapper_timer