class Params:
    def __init__(self,param):
        for key, value in param.items():
            setattr(self,key,value)

    def merge(self,other):
        self.__dict__.update(other.__dict__)
        return self