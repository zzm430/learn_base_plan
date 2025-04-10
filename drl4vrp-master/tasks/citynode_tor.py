import torch
import math
from enum import Enum
from typing import Optional,Dict
from dataclasses import dataclass
from shapely.geometry import Polygon, LineString, Point,MultiPoint,MultiLineString,GeometryCollection

from learn_base_plan.CNN.test_cnn import batch_size
from learn_base_plan.coverage_path_dqn.test_polygon import find_intersections

@staticmethod
def find_intersection(line1: LineString, line2: LineString):
    if line1.intersection(line2):
        intersection = line1.intersection(line2)
        if intersection.geom_type == 'Point':
            return {"intersect_flag": True, "point": (intersection.x, intersection.y)}
    else:
        return {"intersect_flag": False, "point": (0, 0)}

class NodeType(Enum):
    ENDPOINT = "Endpoint"
    DUMMY = "Dummy"

    @classmethod
    def to_tensor(cls,values):
        mapping = {cls.ENDPOINT:0,cls.DUMMY:1}
        return torch.tensor([mapping[v] for v in values])


class CityNodesBatch:
    positions: torch.Tensor
    directions: torch.Tensor
    node_types: torch.Tensor

    previous_positions: torch.Tensor
    real_track_ids: torch.Tensor
    track_lengths: torch.Tensor

    def batch_size(self):
        return self.positions.shape[0] if len(self.positions.shape) == 3 else 1

    def num_nodes(self):
        return self.positions.shape[-2]


class VectorizedCityNodeSystem:
    def __init__(self,extra_cost=5.0,dummy_cost=10.0):
        self.extra_cost = extra_cost
        self.dummy_cost = dummy_cost

    def create_batch_from_nodes(self,tour:torch.Tensor) -> CityNodesBatch:
        """"
        从tour 张量创建CityNodesBatch
        输入:tour [1,20,2] (假设batch_size=1)
        输出:CityNodesBatch 包含所有节点信息
        """
        if( len(tour.shape) ==2):
            tour = tour.unsqueeze(0)

        batch_size,num_nodes,_ = tour.shape
        device = tour.device

        #1.计算节点对关系
        real_track_ids = torch.arange(num_nodes // 2,
                                      device=device).repeat_interleave(2).unsqueeze(0)

        #2.计算方向（atan2(y_diff,x_diff)）
        #获取每对的起点和终点 [batch,num_pairs,2]
        starts = tour[:,::2]   # 偶数索引节点(0,2,4,...)
        ends = tour[:, 1::2]   # 奇数索引节点(1,3,5,...)

        #计算方向向量[batch,num_pairs,2]
        dir_vectors = ends - starts

       #计算角度[batch,num_pairs]
        directions = torch.atan2(dir_vectors[...,1],dir_vectors[...,0])

        #计算轨道长度[batch,num_pairs]
        track_lengths = torch.norm(dir_vectors,dim=-1)

        #长度
        full_track_lengths = torch.zeros(batch_size,num_nodes,device=device)
        full_track_lengths[:,::2] = track_lengths
        full_track_lengths[:,1::2] = track_lengths

        #计算每个节点的前一个节点位置
        prev_positions = torch.zeros_line(tour)
        prev_positions[:,::2] = tour[:,1::2]
        prev_positions[:,1::2] = tour[:,::2]


        #节点类型(all->ENDPOINT)
        node_types = torch.zeros(batch_size,num_nodes,dtype=torch.long,device=device)


        return CityNodesBatch(
            positions=tour,
            directions=directions,
            node_types=node_types,
            prev_positions=prev_positions,
            real_track_ids=real_track_ids,
            track_lengths=full_track_lengths
        )

    def compute_reward_matrix(self,nodes_batch:CityNodesBatch)->torch.Tensor:
        pos = nodes_batch.positions
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)

        batch_size,num_nodes,_ = pos.shape

        #创建索引网格 [batch_size,num_nodes,num_nodes]
        i_idx = torch.arange(num_nodes,device=pos.device).view(1,-1,1).expand(batch_size,num_nodes,num_nodes)
        j_idx = torch.arange(num_nodes,device=pos.device).view(1,1,-1).expand(batch_size,num_nodes,num_nodes)

        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        #base_cost维度 [batch_size,num_nodes,num_nodes]
        base_cost = torch.norm(pos_diff,dim=-1)

        #初始化奖励矩阵
        reward_matrix = torch.full((batch_size,num_nodes,num_nodes),float('-inf'),device=pos.device)

        #对角线设置为NAN
        diag_mask = i_idx == j_idx
        reward_matrix[diag_mask] = torch.nan

        #处理ENDPOINT节点对
        is_endpoint = nodes_batch.node_types == 0
        both_endpoint = is_endpoint.unsqueeze(-1) & is_endpoint.unsequeeze(-2)

        #相同轨道ID的情况
        same_track = (nodes_batch.real_track_ids.unsqueeze(-1) == (nodes_batch.real_track_ids.unsqueeze(-2)))
        same_track_mask = both_endpoint & same_track
        reward_matrix[same_track_mask] = 0

        #不同轨道ID的情况
        diff_track_mask = both_endpoint & ~same_track

        #相同方向
        same_dir = (nodes_batch.directions.unsqueeze(-1) == nodes_batch.directions.unsqueeze(-2))

        same_dir_mask = diff_track_mask & same_dir

        #基础成本+extra_cost
        cost = -base_cost - self.extra_cost

        #track_diff=1
        track_diff = torch.abs(nodes_batch.real_track_ids.unsqueeze(-1) - nodes_batch.real_track_ids.unsqueeze(-2))
        track_diff_1_mask = same_dir_mask & (track_diff == 1)
        cost[track_diff_1_mask] -= self.extra_cost

        #垄长度短并且ID相差<=5
        length_sum = nodes_batch.track_lengths.unsqueeze(-1) + nodes_batch.track_lengths.unsqueeze(-2)
        short_track_mask = same_dir_mask & (length_sum < 100) & (track_diff <=5 )
        cost[short_track_mask] -= self.extra_cost

        reward_matrix[same_dir_mask] = cost[same_dir_mask]

        #不同方向
        diff_dir_mask = diff_track_mask & ~same_dir
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(i):   #只处理下三角
                    if diff_dir_mask[b,i,j]:
                       line1_start = nodes_batch.positions[b,i].cpu().numpy()
                       line1_end = nodes_batch.previous_positions[b,i].cpu().numpy()
                       line2_start = nodes_batch.positions[b,j].cpu().numpy()
                       line2_end = nodes_batch.previous_positions[b,j].cpu().numpy()

                       #创建LineString对象
                       line_one = LineString([line1_start,line1_end])
                       line_two = LineString([line2_start,line2_end])

                       #检查相交
                       result = find_intersections(line_one,line_two)

                       if result["intersect_flag"]:
                           # 计算两节点间距离
                            dis_intersec_pt = torch.norm(nodes_batch.positions[b,i] - nodes_batch.positions[b,j])
                            cost_val = -base_cost[b,i,j] - (5 * dis_intersec_pt +5)
                       else:
                           #计算角度差
                            vec1 = nodes_batch.previous_positions[b,i] - nodes_batch.positions[b,i]
                            vec2 = nodes_batch.previous_positions[b,j] - nodes_batch.positions[b,j]
                            angle_diff = torch.atan2(vec1[1],vec1[0] - torch.atan2(vec2[1],vec2[0]))
                            angle_diff = torch.abs(angle_diff * 180 / math.pi)

                            #计算节点间距离
                            node_distance = torch.norm(nodes_batch.positions[b,i] - nodes_batch.positions[b,j])
                            if angle_diff < 80 and node_distance < 3:
                                cost_val = -base_cost[b,i,j] - self.extra_cost
                            else:
                                cost_val = -base_cost[b,i,j]

                       reward_matrix[b,i,j] = cost_val

        #处理DUMMY节点
        is_dummy = nodes_batch.node_types == 1
        any_dummy = is_dummy.unsqueeze(-1) | is_dummy.unsqueeze(-2)
        reward_matrix[any_dummy] = -self.dummy_cost


        #对称赋值上三角部分
        upper_mask = i_idx < j_idx
        reward_matrix[upper_mask] = reward_matrix.transpose(1,2)[upper_mask]

        return reward_matrix.squeeze(0) if batch_size == 1 else reward_matrix


