import os
import importlib


def get_method(method_name):
    if not os.path.exists("methods/"+method_name+".py"):
        raise ValueError("method does not exist!")
    module = "methods."+method_name
    method_class = getattr(importlib.import_module(module), method_name)
    return method_class
