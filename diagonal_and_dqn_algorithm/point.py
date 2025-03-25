class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def distance_to(self,other):
        if not isinstance(other,Point):
            raise TypeError("other 必须是一个 Point 对象")
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) **0.5