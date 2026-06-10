# 假设是轨迹规划的场景，计算回归损失
import torch
import torch.nn.functional as F 
 
class RegressionLoss(torch.nn.Module):
    def __init__(self, reduction =  'MAE'):
        super().__init__()
        self.reduction =  reduction
        self.delta = 1.0
    def CalculateTrajMAE(self, traj_gt, traj):
        # traj (batch, agent_num, steps, 2)
        mae = abs(traj - traj_gt).mean()
        return mae
    def CalculateTrajMSE(self,traj_gt, traj):
        mse = (0.5*(traj - traj_gt)**2).mean()
        return mse
    def CalculateTrajHuber(self,traj_gt, traj):
        abs_err = abs(traj - traj_gt)
        #l1 region
        l1_region = abs_err > self.delta
        l2_region = ~l1_region
        abs_err[l2_region] = 0.5 * (abs_err[l2_region])**2 / self.delta
        abs_err[l1_region] = (abs_err[l1_region]) - self.delta * 0.5
        hub_loss =  abs_err.mean()
        # hub_loss =  torch.where(l2_region,
        #                         (0.5* (abs_err)**2/ self.delta),
        #                         (abs_err) - self.delta * 0.5)
        # hub_loss =  hub_loss.mean()
        return hub_loss
    def forward(self, loss, traj_gt, traj):
        if self.reduction == 'MAE':
            return self.CalculateTrajMAE(self, traj_gt, traj)
        elif self.reduction == 'MSE':
            return self.CalculateTrajMSE(self, traj_gt, traj)
        elif self.reduction == 'HUBER':
            return self.CalculateTrajHuber(self,traj_gt, traj)

class LaplaceNLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, target, pred):
        # 这里的pred最后维度多两个分量(x,y,bx,by) 预测位置(均值)，及其不确定度
        # pred (batch, agent_num, steps, 4)
        # targ (batch, agent_num, steps, 2)
        scale = pred[:,:,:,0:2]
        loc = pred[:,:,:,2:4] # 拆分后两个维度
        lap_nll = torch.log(scale) + abs(target - loc)/scale
        lap_nll = lap_nll.mean()
        # 考虑预测 log(scale) 而不是 scale
        return lap_nll
if __name__ == '__main__':
    batch_size = 32
    agent_num = 10
    steps = 50
    modal_num = 6
    coord_dim = 2
    trajs = torch.randn(batch_size, 
                        agent_num, 
                        steps, 
                        modal_num,
                        coord_dim)
    
    # winner take all (32,10,50,6,2)
    gt = torch.randn(batch_size,
                     agent_num,
                     steps,
                     coord_dim)
    gt = gt.unsqueeze(3)
    #选择norm = 2 误差最小的模态，消失一个维度.(32,10,50,6)
    l2_norm =torch.norm(trajs - gt, p = 2, dim = -1).sum(dim = -1)
    l2_norm.sum(dim = 2)# 时间上和 (32,10,6)
    best_mode = l2_norm.argmin(dim = -1)# 32,10
#     best_mode 的形状是 (batch, agent)，所以它的取值由 batch 和 agent 
#   ▎ 共同决定。要正确使用它，就必须用 arange 把 batch、agent 的坐标"喂"给它配对
    batch_index = torch.arange(batch_size).view(-1,1)
    agent_index = torch.arange(agent_num).view(1,-1)
    best_traj = trajs[batch_index,agent_index,:,best_mode,:]
    


      

      