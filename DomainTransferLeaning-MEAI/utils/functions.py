from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # output = grad_output.neg() * ctx.alpha
        # output = grad_output.neg() * ctx.alpha*0 #@jinhui 放弃修改 理由：查看和浩鹏师兄21.02.26 20:00的讨论
        output = grad_output.neg() * ctx.alpha #@jinhui 为了尝试学习领域特有信息
        return output, None