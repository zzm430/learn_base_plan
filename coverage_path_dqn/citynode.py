from enum import Enum

class NodeType(Enum):
     ENDPOINT = "Endpoint"
     DUMMY = "Dummy"


class CityNode:
    # 城市节点信息
    def __init__(self,
                 position:tuple,
                 direction:tuple,
                 node_type:NodeType,
                 previous_node,
                 real_track_id ,
                 current_track_length ):
        self.position = position
        self.direction = direction
        self.node_type = node_type
        self.previous_node = previous_node
        self.real_track_id = real_track_id
        self.current_track_length = current_track_length

