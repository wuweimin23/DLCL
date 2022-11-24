"""
Differentiable layer for constrained Lasso problem
"""
import torch
import torch.nn as nn
import differentiable_layer.con_lasso_layer as con_lasso_layer

class Layer(nn.Module):
  def __init__(self, config_model: dict) -> None:
    super().__init__()
    config = {
      "hparams": {
              "dim": 50,  #the dimension of variable in optimization problem
              "max_abs": 0.2,  #the maximum of absolute value in constraints of optimization problem
              "lr": 1e-4,  #learning rate
              "lam": 0.1,  #regularization parameter for Lasso problem
              "para_method_Sigma": 1,  #the index for parameterized method for Sigma in optimization problem
              "para_method_b": 1  #the index for parameterized method for b in optimization problem
              },
      "name": "Layer",
    }
  
    self.name = config["name"]
  
    config["hparams"].update(config_model["hparams"])
    hparams = config['hparams']
    self.dim = hparams['dim']
    self.max_abs = hparams['max_abs']
    self.lr = hparams['lr']
    self.lam = hparams['lam']
    self.para_method_Sigma = hparams['para_method_Sigma']
    self.para_method_b = hparams['para_method_b']
    
    #Generating the parameter matrix in optimization problem
    if self.para_method_Sigma == 1:
      self.W1 = nn.Parameter(torch.zeros((self.dim, self.dim)))
      self.W2 = nn.Parameter(torch.zeros(self.dim, self.dim))
    elif self.para_method_Sigma == 2:
      self.W1 = nn.Parameter(torch.zeros((self.dim, self.dim)))
    else:
      self.W2 = nn.Parameter(torch.zeros(self.dim, self.dim))
      
    if self.para_method_b == 1:
      self.W3 = nn.Parameter(torch.zeros(self.dim, self.dim))
      self.W4 = nn.Parameter(torch.zeros(self.dim, 1))
    elif self.para_method_b == 2:
      self.W4 = nn.Parameter(torch.zeros(self.dim, 1))
    elif self.para_method_b == 3:
      self.W3 = nn.Parameter(torch.zeros(self.dim, self.dim))
    
    #define matrix for constraints
    #(1). sum(x) = 0
    self.B = torch.ones(1, self.dim)
    self.c = torch.zeros((1))
    
    #(2). |x[1,:]| \leq max_abs
    vector = torch.ones((self.dim-1))
    D = torch.diag(vector, 1)[:(self.dim-1),:]
    self.D = torch.cat((D, -D), dim=0)
    self.g = torch.ones((2*(self.dim-1))) * self.max_abs
        
  def loss(self, x, output, I):
    result = torch.matmul(output, x) - I
    loss = torch.sum(result*result)
    return loss

  def forward(self, Sigma =None, b = None, Sigma_cholesky = None, output = None):
    #Sigma is the covariance matrix at time t, shape: (number_batch, batch_size, n, n)
    #b: shape: (number_batch, batch_size, n)
    #Sigma_cholesky is a lower triangular matrix
    #output is the covariance matrix at time t
    
    device = Sigma.device
    number_batch = Sigma.shape[0]
    batch_size = Sigma.shape[1]
    if self.para_method_Sigma == 1:
      Sigma_cholesky_para = Sigma_cholesky + torch.einsum('jk, hikl->hijl', self.W1, Sigma_cholesky) + self.W2
    elif self.para_method_Sigma == 2:
      Sigma_cholesky_para = Sigma_cholesky + torch.einsum('jk, hikl->ijl', self.W1, Sigma_cholesky) 
    else:
      Sigma_cholesky_para = Sigma_cholesky + self.W2
    S_hat = torch.einsum('hijk, hikl->hijl', Sigma_cholesky_para, Sigma_cholesky_para.transpose(2,3))
    
    if self.para_method_b == 1:
      b_hat = b + torch.einsum('jk, hik->hij', self.W3, b) + self.W4
    elif self.para_method_b == 2:
      b_hat = b + self.W4
    elif self.para_method_b == 3:
      b_hat = b + torch.einsum('jk, hik->hij', self.W3, b)
    else: 
      b_hat = b
      
    #constraints
    self.B = self.B.to(device)
    self.c = self.c.to(device)
    self.D = self.D.to(device)
    self.g = self.g.to(device)
    B = self.B.unsqueeze(0).unsqueeze(0).repeat(number_batch,batch_size,1,1)
    c = self.c.unsqueeze(0).unsqueeze(0).repeat(number_batch,batch_size,1)
    D = self.D.unsqueeze(0).unsqueeze(0).repeat(number_batch,batch_size,1,1)
    g = self.g.unsqueeze(0).unsqueeze(0).repeat(number_batch,batch_size,1)
    
    x = con_lasso_layer.Layer(lam = self.lam)(S_hat, b_hat, B, c, D, g).transpose(1,2)

    I = torch.eye(self.dim).to(device)
    loss = self.loss(x, output, I)
    
    pred = x

    return pred, loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)