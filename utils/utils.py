import torch
import data_loader.data_loader as data_loader


class Trainer():
  def __init__(self, model, data, device, batch_size = 2, max_epochs = 1, ite_val = 100):
    self.model = model.to(device)
    self.data = data
    self.device = device
    self.batch_size = batch_size
    self.max_epochs = max_epochs
    self.stop = False
    self.eps = 1e-5
    self.ite_val = ite_val #the number of training iterations for each validation step
    
  def prepare_data(self):
    self.train_data, self.val_data, self.test_data = data_loader.dl(self.data, self.device, self.batch_size)

  def fit(self):
    self.optim = self.model.configure_optimizers()
    self.epoch = 0
    for self.epoch in range(self.max_epochs):
      print("-----------------epoch = {0}".format(self.epoch))
      self.fit_epoch()

  def fit_epoch(self):
    loss_val_criterion = 1000000000000
    self.model.train()
    #cum_loss is the training loss
    cum_loss = torch.tensor(0.0)
    for i, (Sigma,b,L,output) in enumerate(self.train_data):
      pred, loss = self.model(Sigma,b,L,output)
      cum_loss += loss.item()
      if i >= 0:
        print(f"training loss = {cum_loss / (i + 1):.4f}")
      self.optim.zero_grad()
      loss.backward()
      self.optim.step()

      if i > 0 and i % (self.ite_val) == 0:
        self.model.eval()
        cum_loss_val = torch.tensor(0.0)
        for j, (Sigma,b,L,output) in enumerate(self.val_data):
          pred, loss= self.model(Sigma,b,L,output)
          cum_loss_val += loss.item()
        
        loss_val = cum_loss_val / (j+1)
        print(f"----------------validation loss = {loss_val:.4f}")
        self.model.train()
        #log the parameters of model
        if loss_val < loss_val_criterion:
          loss_val_criterion = loss_val
          torch.save(self.model.state_dict(),'./results/model.pkl')
          
  
  def test(self, load_state = True):
    self.model.eval()
    if load_state == True:
      self.model.load_state_dict(torch.load('./results/model.pkl'))
    cum_loss_test = torch.tensor(0.0)
    with torch.no_grad():
      for j, (Sigma,b,L,output) in enumerate(self.test_data):
        pred, loss = self.model(Sigma,b,L,output)
        cum_loss_test += loss.item()
      print(f"---------------test loss = {cum_loss_test / (j+1):.4f}")

