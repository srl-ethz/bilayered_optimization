import torch
import torch.nn as nn
import torch.autograd as autograd

from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector
from py_diff_pd.common.common import ndarray


class SimFunction(autograd.Function):

    @staticmethod
    def forward(ctx, deformable, dofs, act_dofs, method, q, v, a, f_ext, dt, option, w=None):

        ctx.deformable = deformable
        ctx.dofs = dofs
        ctx.act_dofs = act_dofs
        ctx.method = method
        ctx.dt = dt
        ctx.option = option

        ctx.w_requires_grad = w is not None

        q_array = StdRealVector(q.detach().numpy())
        v_array = StdRealVector(v.detach().numpy())
        a_array = StdRealVector(a.detach().numpy())
        f_ext_array = StdRealVector(f_ext.detach().numpy())
        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)

        deformable.PyForward(
            method, q_array, v_array, a_array, f_ext_array, dt, option, q_next_array, v_next_array, StdIntVector(0))

        q_next = torch.as_tensor(ndarray(q_next_array))
        v_next = torch.as_tensor(ndarray(v_next_array))

        ctx.save_for_backward(q, v, a, f_ext, q_next, v_next)

        return q_next, v_next

    @staticmethod
    def backward(ctx, dl_dq_next, dl_dv_next):

        q, v, a, f_ext, q_next, v_next = ctx.saved_tensors
        dofs, act_dofs = ctx.dofs, ctx.act_dofs

        w_requires_grad = ctx.w_requires_grad

        q_array = StdRealVector(q.detach().numpy())
        v_array = StdRealVector(v.detach().numpy())
        a_array = StdRealVector(a.detach().numpy())
        f_ext_array = StdRealVector(f_ext.detach().numpy())
        q_next_array = StdRealVector(q_next.detach().numpy())
        v_next_array = StdRealVector(v_next.detach().numpy())

        dl_dq_next_array = StdRealVector(dl_dq_next.detach().numpy())
        dl_dv_next_array = StdRealVector(dl_dv_next.detach().numpy())

        w_dofs = ctx.deformable.NumOfPdElementEnergies()
        num_elements = ctx.deformable.NumOfElements()

        dl_dq_array = StdRealVector(dofs)
        dl_dv_array = StdRealVector(dofs)
        dl_da_array = StdRealVector(act_dofs)
        dl_df_ext_array = StdRealVector(dofs)
        dl_dwi = StdRealVector(w_dofs * num_elements)
        dl_dstate_p = StdRealVector(dofs)

        ctx.deformable.PyBackward(
            ctx.method, q_array, v_array, a_array, f_ext_array, ctx.dt,
            q_next_array, v_next_array, StdIntVector(0), dl_dq_next_array, dl_dv_next_array, ctx.option,
            dl_dq_array, dl_dv_array, dl_da_array, dl_df_ext_array, dl_dwi, dl_dstate_p)

        dl_dq = torch.as_tensor(ndarray(dl_dq_array))
        dl_dv = torch.as_tensor(ndarray(dl_dv_array))
        dl_da = torch.as_tensor(ndarray(dl_da_array))
        dl_df_ext = torch.as_tensor(ndarray(dl_df_ext_array))

        dl_dw = torch.as_tensor(ndarray(dl_dwi).reshape(w_dofs, num_elements)) if w_requires_grad else None

        return (None, None, None, None,
            torch.as_tensor(ndarray(dl_dq)),
            torch.as_tensor(ndarray(dl_dv)),
            torch.as_tensor(ndarray(dl_da)),
            torch.as_tensor(ndarray(dl_df_ext)),
            None, None, dl_dw)



class Sim(nn.Module):
    def __init__(self, deformable):
        super(Sim, self).__init__()
        self.deformable = deformable

    def forward(self, dofs, act_dofs, method, q, v, a, f_ext, dt, option, w=None):
        return SimFunction.apply(
            self.deformable, dofs, act_dofs, method, q, v, a, f_ext, dt, option, w)