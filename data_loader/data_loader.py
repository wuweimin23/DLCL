import torch.utils.data as Data

#data loader
def dl(data, device, batch_size = 1):
  #data is a dictionary, includes four kinds of data, and data[key[i]] is a tensor
  key = list(data.keys())
  data1 = data[key[0]].to(device)
  data2 = data[key[1]].to(device)
  data3 = data[key[2]].to(device) 
  data4 = data[key[3]].to(device) 
  
  
  size = data1.shape[0]
  #use 80% of data for training set, 10% for validating set, 10% for testing set
  train_end_index = int(0.8 * size)
  val_end_index = int(0.9 * size)
  
  torch_dataset_train = Data.TensorDataset(data1[:train_end_index], data2[:train_end_index], data3[:train_end_index], data4[:train_end_index])
  loader_train = Data.DataLoader(
      dataset = torch_dataset_train,
      batch_size = batch_size,
      shuffle = True,
      num_workers = 0,
      drop_last = True,
  )

  torch_dataset_val = Data.TensorDataset(data1[train_end_index:val_end_index], data2[train_end_index:val_end_index], data3[train_end_index:val_end_index], data4[train_end_index:val_end_index])
  loader_val = Data.DataLoader(
      dataset = torch_dataset_val,
      batch_size = batch_size,
      shuffle = True,
      num_workers = 0,
      drop_last = True,
  )

  torch_dataset_test = Data.TensorDataset(data1[val_end_index:], data2[val_end_index:], data3[val_end_index:], data4[val_end_index:])
  loader_test = Data.DataLoader(
      dataset = torch_dataset_test,
      batch_size = batch_size,
      shuffle = True,
      num_workers = 0,
      drop_last = True,
  )
  
  return loader_train,loader_val,loader_test