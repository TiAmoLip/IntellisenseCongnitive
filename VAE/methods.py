import torch
from torchvision.utils import save_image
from tqdm import tqdm
import os


class EarlyStopping:
    def __init__(self,config, patience=7, verbose=False, delta=-0.01):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.config = config
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1000
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if score < self.best_score:        
                self.best_score = score
                self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        name = 'checkpoint'
        for key in self.config.keys():
            name += '_'+str(key)+'_'+str(self.config[key])
        torch.save(model.state_dict(), f'./model/{name}.pth')
        self.val_loss_min = val_loss
        
def train(model, train_loader, current_epoch, total_epoch,device, optimizer,config,f):
    model.train()
    name = ''
    for key in config.keys():
        name += '_'+str(key)+'_'+str(config[key])
    with tqdm(total = len(train_loader)) as pbar:
        # for epoch in range(batch_epoch):
            for batch_idx, (data, _) in enumerate(train_loader):
                # print ("In train stage: data size: {}".format(data.size()))
            
                data = data.to(device)
                optimizer.zero_grad()
                loss = model.loss_func(data)
                loss.backward()
                optimizer.step()
                if (batch_idx+1) % 100 == 0:
                    # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx, len(train_loader), loss.item()))
                    pbar.set_description('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(current_epoch+1, total_epoch, batch_idx, len(train_loader), loss.item()))
                    f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'.format(current_epoch+1, total_epoch, batch_idx, len(train_loader), loss.item()))
                    pbar.update(100)
                    f.flush()
                if batch_idx == 0:
                    # nelem = data.size(0)
                    sampled = model.sample(50)
                    nrow  = 10
                    os.makedirs("images/"+name, exist_ok=True)
                    save_image(sampled.view(sampled.shape[0], 1, 28, 28), f'./images/{name}/{current_epoch}' + '.png', nrow=nrow)
# iterator over test set
def test(model, test_loader,device,config):
    val_loss = 0
    cnt = 0
    name = ''
    for key in config.keys():
        name += '_'+str(key)+'_'+str(config[key])
    with torch.no_grad():
        model.eval()
        with tqdm(total = len(test_loader)) as pbar:
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                loss = model.loss_func(data,only_reconstruction=True)
                # print ("In test stage: data size: {}, loss: {}".format(data.size(), loss.item()))
                cnt+=data.shape[0]
                val_loss += loss.item()*data.shape[0]
                if batch_idx == 0:
                    sampled = model.sample(50)
                    nrow  = 10
                    save_image(sampled.view(sampled.shape[0], 1, 28, 28), f'./images/{name}/image_test' + '.png', nrow=nrow)
                pbar.set_description('Test Step [{}/{}], Loss: {:.4f}'.format(batch_idx, len(test_loader), loss.item()))
                pbar.update(1)
    return val_loss/cnt


def run_training_session(model, train_loader, test_loader, config ,optimizer, device,continue_training_path = None, eval_freq=3):
    early_stopping = EarlyStopping(patience=7, verbose=True,config=config)
    if continue_training_path is not None:
        try:
            model.load_state_dict(torch.load(continue_training_path))
        except:
            print ("Error loading model from path: {}".format(continue_training_path))
    
    with open(f"logs/latent_{model.latent_size}_p_{model.p}_layers_{model.num_layers}_base_channels_{model.base_channels}_lambda_kl_{model.lambda_kl}_lr{optimizer.param_groups[0]['lr']}.txt", "w") as f:
        for i in range(100):
            train(model, train_loader, current_epoch=i, total_epoch=100,device=device, optimizer=optimizer,config=config, f=f)
            if (i+1)%eval_freq==0:
                val_loss = test(model, test_loader,device,config=config)
                f.write(f'Epoch {i+1}, Val Loss: {val_loss}\n')
                f.flush()
                early_stopping(val_loss, model)
            if early_stopping.early_stop:
                f.write('Early stopping\n')
                break
        