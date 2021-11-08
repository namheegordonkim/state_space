import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from train import EnvFactory

class MLP(nn.Module):

    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim,2),
            nn.ReLU(),
            nn.Linear(2,output_dim)
        )

    def forward(self,x):
        return self.sequential(x)


class StateDataset(Dataset):

    def __init__(self):
        self.x,self.y = self.create_dataset()

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

    def create_dataset(self):

        policy_path = 'policies/envs:Car1DEnv-v1/ppo/16_16/latest_16_16.zip' # args.policy_path
        stats_path = 'policies/envs:Car1DEnv-v1/ppo/16_16/latest_stats_16_16.pth'
        expert = PPO.load(policy_path)
        factory = EnvFactory('envs:Car1DEnv-v1')
        env = DummyVecEnv([factory.make_env])
        env = VecNormalize.load(stats_path, env)
        env.training = False

        states = []
        for i in np.arange(-10, 110):
            for j in np.arange(-3, 3, 0.05):
                states.append([i, j])
        states = np.stack(states)
        states_scaled = env.normalize_obs(states)
        states_tensor = torch.as_tensor(states_scaled).float()
        print(states_tensor.shape)

        policy: ActorCriticPolicy = expert.policy.cpu()
        true_actions_tensor, _, _ = policy.forward(states_tensor, deterministic=True)

        return states_tensor,true_actions_tensor

def train(n_epochs,model,optimizer,loss_fn,dataloader):

    for epoch in range(1,n_epochs+1):
        print(epoch)

        epoch_loss = []
        for states,actions in dataloader:
            
            y_pred = model(states)
            loss_train = loss_fn(y_pred,actions)
            
            epoch_loss.append(loss_train.item())
            optimizer.zero_grad()
            loss_train.backward() # retain_graph?
            optimizer.step()
        
        print(f'Epoch {epoch}, Training loss {np.mean(epoch_loss):.4f}')
        print()

def main():

    dataset = StateDataset()
    dataloader = DataLoader(dataset=dataset,batch_size=16,shuffle=True)
    model = MLP(input_dim=2,output_dim=1)
    optimizer = optim.SGD(model.parameters(),lr=1e-2)
    trained_model = train(
        n_epochs = 10,
        model = model,
        optimizer = optimizer,
        loss_fn = nn.MSELoss(),
        dataloader = dataloader
    )

    torch.save(trained_model.state_dict(),'state_dict.zip')


if __name__ == "__main__":
    main()