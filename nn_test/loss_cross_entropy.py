import torch
import torch.nn.functional as F 

class CrossEntroyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def CalculateCELossByIndexGT(self, target, pred_scores):
        # target (bs,) 正确类别索引
        # pred_scores (bs, n) 对n种类各自的分数
        batch_size = pred_scores.shape[0]

        # ===== 版本 1：分步（教学清晰）=====
        # cls_prob = F.softmax(pred_scores, dim=-1)  # (bs, n) 概率
        # log_prob = torch.log(cls_prob)  # (bs, n) 取对数（注意：这里不取负）
        # batch_index = torch.arange(batch_size)
        # selected_log_probs = log_prob[batch_index, target]  # (bs,) 挑出每个样本对应真实类的 log(概率)
        # loss = -selected_log_probs  # (bs,) 取负
        # return loss.mean()

        # ===== 版本 2：优化（数值稳定，log_softmax = softmax + log 两步合并）=====
        cls_prob = F.log_softmax(pred_scores, dim=-1)  # (bs, n) 返回 log(概率)
        batch_index = torch.arange(batch_size)
        selected_log_probs = cls_prob[batch_index, target]  # (bs,) 挑出每个样本对应真实类的 log(概率)
        loss = -selected_log_probs  # (bs,) 取负，交叉熵 = -log(p)
        return loss.mean()  # 求平均

    def CalculateCELossByOnehot(self, one_hot, pred_scores):
        # one_hot: (bs, n) one-hot 编码，如 [0, 1, 0] 表示第 1 类
        # pred_scores: (bs, n) 对 n 种类各自的分数

        # ===== 版本 1：分步（教学清晰）=====
        # probs = F.softmax(pred_scores, dim=-1)  # (bs, n) 概率
        # log_prob = torch.log(probs)  # (bs, n) 取对数
        # loss = -(one_hot * log_prob).sum(dim=-1)  # (bs,) 逐元素乘，沿类别维求和
        # return loss.mean()

        # ===== 版本 2：优化（数值稳定）=====
        log_prob = F.log_softmax(pred_scores, dim=-1)  # (bs, n) log(概率)
        loss = -(one_hot * log_prob).sum(dim=-1)  # (bs,) 逐元素乘并求和
        return loss.mean()

    def CalculateBCELoss(self, target, scores):
        # target: (batch_size,) 二分类标签，0 或 1
        # scores: (batch_size,) 原始 logits（未经 sigmoid）

        # ===== 版本 1：分步（教学清晰）=====
        # probs = torch.sigmoid(scores)  # (bs,) 正类概率
        # # 正样本（target=1）贡献：-log(p)
        # # 负样本（target=0）贡献：-log(1-p)
        # * target 只是组合起来而已， 一个公式表达方便
        # loss = -(target * torch.log(probs) + (1 - target) * torch.log(1 - probs))
        # return loss.mean()

        # ===== 版本 2：优化（数值稳定，BCEWithLogitsLoss 风格）=====
        probs = torch.sigmoid(scores)  # (bs,) 正类概率
        # BCE = -(y·log(p) + (1-y)·log(1-p))
        loss = -(target * torch.log(probs) + (1 - target) * torch.log(1 - probs))
        return loss.mean() 
        

if __name__ == "__main__":
    # 测试：和 PyTorch 官方的 CrossEntropyLoss 对比
    batch_size = 4
    num_classes = 3

    # 模拟数据
    pred_scores = torch.randn(batch_size, num_classes)
    target = torch.tensor([0, 2, 1, 0])  # 真实类别索引

    # 我们的实现
    loss_fn = CrossEntroyLoss()
    our_loss = loss_fn.CalculateCELossByIndexGT(target, pred_scores)

    # PyTorch 官方实现
    official_loss_fn = torch.nn.CrossEntropyLoss()
    official_loss = official_loss_fn(pred_scores, target)

    print("我们的实现:", our_loss.item())
    print("官方实现:  ", official_loss.item())
    print("差值:      ", abs(our_loss.item() - official_loss.item()))
    print("是否一致:  ", torch.allclose(our_loss, official_loss))

