import taichi as ti

@ti.data_oriented
class BaseClass(object):
    @property
    def one_property(self):
        return "Property of super-class"
    @ti.kernel
    def compute(self):
        pass
    @ti.func
    def some_function(self):
        pass

class DeviatedClass(BaseClass): 
    @property
    def one_property(self):
        return "Property of sub-class"


a = DeviatedClass()
print(a.one_property)
# output: Property of super-class
