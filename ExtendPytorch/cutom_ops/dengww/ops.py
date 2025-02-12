import torch
from torch import Tensor

__all__ = ["mymuladd"]

def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.dengww.mymuladd.default(a, b, c)

# 注册一个 FakeTensor 内核（又名“元内核”、“抽象实现”）
# 描述输出 Tensor 的属性
# 输入 Tensor 的属性。FakeTensor 内核是必需的
# 以便 op 能够高效地与 torch.compile 配合使用。
@torch.library.register_fake("dengww::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)

def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.dengww.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.dengww.mymul.default(grad, a)
    return grad_a, grad_b, None

def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)

# 此代码为算子添加了训练支持。您必须向我们提供
# 算子的反向公式和一个 `setup_context` 函数
# 以保存要在反向中使用的值。
torch.library.register_autograd(
    "dengww::mymuladd", _backward, setup_context=_setup_context
)

@torch.library.register_fake("dengww::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)

def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.dengww.myadd_out.default(a, b, out)