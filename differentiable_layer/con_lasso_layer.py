import torch
import torch.autograd as auto
import math

"""A differentiable constrained Lasso optimization layer

  A Constrained Lasso Layer solves a parametrized constrained Lasso optimization problem.
  It solves the problem in its forward pass by ADMM method, and it computes 
  the derivative of problem's solution map with respect to the parameters in
  its backward pass. 

  Constrained Lasso Problem
  argmin_{x} 0.5x^{T}Zx + b^{T}x + \lambda |x|_{1}
  s.t. Bx = c
       Dx <= g
        
"""

def Layer(rou=1.0, lam=1.0, eps_abs = 1e-3, eps_rel = 1e-3, step_size = 1.0):
  """Construct a Constrained Lasso Layer

  Args:
    rou: scalar, parameter for penality item in augemented largrangian function
    lam: scalar, parameter for norm1 in constrained Lasso problem
    eps_abs: parameter for absolute error of stopping condition
    eps_rel: parameter for relative error of stopping condition
    step_size: parameter for step size in intertion
    
  """
  class Layer_Fn(auto.Function):
    def forward(ctx, Z, b, B, c, D, g):
      """
      Arguments in constrained LASSO problem as mentioned before
      Z: (number_batch, batch_size, n, n)
      b: (number_batch, batch_size, n)
      B: (number_batch, batch_size, m, n)
      c: (number_batch, batch_size, m)
      D: (number_batch, batch_size, k, n)
      g: (number_batch, batch_size, k)
      """
      ctx.device = Z.device
      #number parameter in problem
      ctx.number_batch = Z.shape[0]
      ctx.batch_size = Z.shape[1]
      ctx.num_var = Z.shape[2]
      ctx.num_neq = B.shape[2]
      ctx.num_ineq = D.shape[2]
      
      #initialize
      #primal variable
      x = torch.zeros((ctx.number_batch, ctx.batch_size, ctx.num_var)).to(ctx.device)
      z = torch.zeros((ctx.number_batch, ctx.batch_size, ctx.num_var)).to(ctx.device)
      s = torch.zeros((ctx.number_batch, ctx.batch_size, ctx.num_ineq)).to(ctx.device)
      #dual variable
      mu = torch.zeros((ctx.number_batch, ctx.batch_size, ctx.num_neq)).to(ctx.device)
      omega = torch.zeros((ctx.number_batch, ctx.batch_size, ctx.num_ineq)).to(ctx.device)
      beta = torch.zeros((ctx.number_batch, ctx.batch_size, ctx.num_var)).to(ctx.device)
        
      inv_tensor = torch.linalg.inv(Z + rou * (torch.eye(ctx.num_var).to(ctx.device) + torch.matmul(B.transpose(2,3), B) + \
          torch.matmul(D.transpose(2,3), D)))
      
      #iteration
      res_pr = 1.0
      res_du = 1.0
      eps_pr = 0.0
      eps_du = 0.0
      I_num_var = torch.eye(ctx.num_var).to(ctx.device).unsqueeze(0).unsqueeze(0).repeat(ctx.number_batch,ctx.batch_size,1,1)
      A_1 = torch.cat((torch.cat((B,D), dim=2), I_num_var),dim=2)
      A_2 = torch.cat((torch.cat((torch.zeros(B.shape).to(ctx.device),torch.zeros(D.shape).to(ctx.device)), dim=2), -I_num_var), dim=2)
      A_3 = torch.cat((torch.cat((torch.zeros((ctx.number_batch, ctx.batch_size,ctx.num_neq,ctx.num_ineq)).to(ctx.device),torch.eye(ctx.num_ineq).to(ctx.device).unsqueeze(0).unsqueeze(0).repeat(ctx.number_batch, ctx.batch_size,1,1)),dim=2), torch.zeros(ctx.number_batch, ctx.batch_size,ctx.num_var,ctx.num_ineq).to(ctx.device)), dim=2)
      E = torch.cat((torch.cat((c,g),dim=2),torch.zeros((ctx.number_batch,ctx.batch_size,ctx.num_var)).to(ctx.device)),dim=2)
      zeros_z = torch.zeros(z.shape).to(ctx.device)
      zeros_s = torch.zeros(s.shape).to(ctx.device)
      
      while ((res_pr > eps_pr) or (res_du > eps_du)):
        x = -b.unsqueeze(-1) + rou*((z-beta).unsqueeze(-1)+torch.matmul(D.transpose(2,3),(g-omega-s).unsqueeze(-1))+\
            torch.matmul(B.transpose(2,3),(c-mu).unsqueeze(-1)))
        x = torch.matmul(inv_tensor, x)
        z_old = z.clone()
        z = x.squeeze(-1) + beta
        z = torch.max((1 - (lam/rou)* torch.reciprocal(torch.abs(z))), zeros_z) * z
        s_old = s.clone()
        s = -torch.matmul(D,x).squeeze(-1) + (g - omega)
        s = torch.max(s, zeros_s)
        mu = mu + rou*step_size*(torch.matmul(B,x).squeeze(-1) - c)
        omega = omega + rou*step_size*(torch.matmul(D, x).squeeze(-1) + s - g)
        x = x.squeeze(-1)
        beta = beta + rou*step_size*(x - z)
        
        #stopping parameter
        res_pr = torch.norm(((torch.matmul(A_1, x.unsqueeze(-1))+torch.matmul(A_2, z.unsqueeze(-1))+torch.matmul(A_3, s.unsqueeze(-1))).squeeze(-1)-E),p=2)
        res_du = torch.norm(rou*(torch.matmul(D.transpose(2,3),(s_old-s).unsqueeze(-1)).squeeze(-1) - (z_old - z)), p=2)
        norm_tensor = torch.tensor((torch.norm(torch.matmul(A_1, x.unsqueeze(-1)), p=2), torch.norm(torch.matmul(A_2, z.unsqueeze(-1)), p=2), torch.norm(torch.matmul(A_3, s.unsqueeze(-1)), p=2), torch.norm(E, p=2)))
        eps_pr = math.sqrt(ctx.num_var+ctx.num_neq+ctx.num_ineq)*eps_abs + eps_rel*torch.max(norm_tensor)
        U = torch.cat((torch.cat((mu,omega),dim=2),beta),dim=2)
        eps_du = math.sqrt(ctx.num_var+ctx.num_neq+ctx.num_ineq)*eps_abs + eps_rel*rou*torch.norm(torch.matmul(A_1.transpose(2,3),U.unsqueeze(-1)),p=2)
      
      #dual variable
      ctx.mu = mu
      ctx.omega = omega
      ctx.save_for_backward(x, Z, b, B, c, D, g)
      
      return x
    
    def backward(ctx, dl_x):
      x, Z, b, B, c, D, g = ctx.saved_tensors
      
      K_1 = torch.cat((torch.cat((Z, D.transpose(2,3)*ctx.omega.unsqueeze(1)), dim=3),B.transpose(2,3)), dim=3)
      K_2 = torch.cat((torch.cat((D, torch.diag_embed(torch.matmul(D,x.unsqueeze(-1)).squeeze(-1)-g)), dim=3), torch.zeros((ctx.number_batch,ctx.batch_size,ctx.num_ineq, ctx.num_neq)).to(ctx.device)), dim=3)
      K_3 = torch.cat((B, torch.zeros((ctx.number_batch, ctx.batch_size, ctx.num_neq, (ctx.num_ineq+ctx.num_neq))).to(ctx.device)), dim=3) 
      K = torch.cat((K_1, K_2), dim=2)
      K = -torch.cat((K, K_3), dim=2)
      
      y = torch.cat((dl_x, torch.zeros((ctx.number_batch, ctx.batch_size, (ctx.num_neq+ctx.num_ineq))).to(ctx.device)), dim=2).unsqueeze(-1)
      
      #solve KKT
      K_LU = torch.lu(K)
      d = torch.lu_solve(y, *K_LU).squeeze(-1)
      d_x = d[:,:, :ctx.num_var]
      d_omega = d[:,:,ctx.num_var:(ctx.num_var+ctx.num_ineq)]
      d_mu = d[:,:,(ctx.num_var+ctx.num_ineq):]
      
      d_Z = 0.5 * (torch.matmul(d_x.unsqueeze(-1), x.unsqueeze(2)) + torch.matmul(x.unsqueeze(-1), d_x.unsqueeze(2)))
      d_c = d_x.clone()
      d_A = torch.matmul(d_mu.unsqueeze(-1), x.unsqueeze(2)) + torch.matmul(ctx.mu.unsqueeze(-1), d_x.unsqueeze(2))
      d_b = -d_mu.clone()
      d_G = torch.matmul(d_omega.unsqueeze(-1), x.unsqueeze(2)) * ctx.omega.unsqueeze(-1) + torch.matmul(ctx.omega.unsqueeze(-1), d_x.unsqueeze(2)) 
      d_h = -d_omega * ctx.omega
      
      grads = (d_Z, d_c, d_A, d_b, d_G, d_h)

      return grads
    
  return Layer_Fn.apply