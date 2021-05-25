import pkgutil
import sys


def load_all_modules_from_dir(dirname):
    for importer, package_name in pkgutil.iter_modules([dirname]):
        full_package_name = '%s.%s' % (dirname, package_name)
        print(full_package_name)
        if full_package_name not in sys.modules:
            module = importer.find_module(package_name
                        ).load_module(full_package_name)
            print(module)

    
load_all_modules_from_dir('msos_project')
