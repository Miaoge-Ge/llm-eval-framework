import os
import pkgutil
import importlib

# Automatically import all modules in this package so they register themselves
pkg_dir = os.path.dirname(__file__)
for module_loader, name, ispkg in pkgutil.iter_modules([pkg_dir]):
    importlib.import_module(f".{name}", __name__)
