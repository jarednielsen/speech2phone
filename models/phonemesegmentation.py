# -*- coding: utf-8 -*-
"""PhonemeSegmentation.ipynb
This is downloaded directly from Jupyter Notebook. DON'T RUN!

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HNiSoHQaphiTQPouOoFEtL5qgVCRxFr4
"""

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable

import time
from os import path
import pickle
import gc

from google.colab import drive
drive.mount('/content/gdrive')

assert torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMIT_root = '/content/gdrive/My Drive/data/TIMIT'
audio_path = path.join(TIMIT_root, "train/audios.pkl")
with open(audio_path, 'rb') as infile:
    audios, segments = pickle.load(infile)

TIMIT_root = '/content/gdrive/My Drive/data/TIMIT'
test_audio_path = path.join(TIMIT_root, "test/audios.pkl")
with open(test_audio_path, 'rb') as infile:
    audios_test, segments_test = pickle.load(infile)

"""# Classifier and Recognizer"""

class PhonemeRecognizer(nn.Module):
    def __init__(self):
        super(PhonemeRecognizer, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 30, stride=10)
        self.conv1_1 = nn.Conv1d(64, 64, 15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(64, 128, 15, stride=2)
        self.conv2_1 = nn.Conv1d(128, 128, 5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 5, stride=2)
        self.conv3_1 = nn.Conv1d(256, 256, 3, stride=1, padding=1)
        self.lstm = nn.LSTM(256, 256, 3, bidirectional=False, dropout=0.1)
        self.linear = nn.Linear(256*3, 256)
        self.linear2 = nn.Linear(256, 1)
    
    def forward(self, x, l):
        x = F.dropout(F.relu(self.conv1(x)), p=0.1)
        x = F.relu(self.conv1_1(x))
        x = F.dropout(F.relu(self.conv2(x)), p=0.1)
        x = F.relu(self.conv2_1(x))
        x = F.dropout(F.relu(self.conv3(x)), p=0.1)
        x = F.relu(self.conv3_1(x))
        factor = l.max().item() / x.size(2)
        new_lengths = l.type(dtype=torch.float) / factor
        new_lengths = new_lengths.type(dtype=torch.int)
        new_lengths[new_lengths==0] = 1
        x = pack_padded_sequence(x.permute(2, 0, 1), new_lengths)
        out, (h_n, c_n) = self.lstm(x)
        h_n = h_n.permute(1, 0, 2).reshape(-1, 1, 3*256)
        y = F.relu(self.linear(h_n))
        y = self.linear2(y).squeeze(1)
        return y
    
class PhonemeClassifier(nn.Module):
    def __init__(self):
        super(PhonemeClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 30, stride=10)
        self.conv1_1 = nn.Conv1d(64, 64, 15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(64, 128, 15, stride=2)
        self.conv2_1 = nn.Conv1d(128, 128, 5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 5, stride=2)
        self.conv3_1 = nn.Conv1d(256, 256, 3, stride=1, padding=1)
        self.lstm = nn.LSTM(256, 256, 3, bidirectional=False, dropout=0.1)
        self.linear = nn.Linear(256*3, 256)
        self.linear2 = nn.Linear(256, 61)
    
    def forward(self, x, l):
        x = F.dropout(F.relu(self.conv1(x)), p=0.1)
        x = F.relu(self.conv1_1(x))
        x = F.dropout(F.relu(self.conv2(x)), p=0.1)
        x = F.relu(self.conv2_1(x))
        x = F.dropout(F.relu(self.conv3(x)), p=0.1)
        x = F.relu(self.conv3_1(x))
        factor = l.max().item() / x.size(2)
        new_lengths = l.type(dtype=torch.float) / factor
        new_lengths = new_lengths.type(dtype=torch.int)
        new_lengths[new_lengths==0] = 1
        x = pack_padded_sequence(x.permute(2, 0, 1), new_lengths)
        out, (h_n, c_n) = self.lstm(x)
        h_n = h_n.permute(1, 0, 2).reshape(-1, 1, 3*256)
        y = F.relu(self.linear(h_n))
        y = self.linear2(y).squeeze(1)
        return y

def load_recognizer(name="recognizer.pt"):
    model = PhonemeRecognizer()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    model.cuda()
    checkpoint = torch.load('/content/gdrive/My Drive/models/' + name)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def load_classifier(name="lstm_model_padding.pt"):
    model = PhonemeClassifier()
    model.cuda()
    checkpoint = torch.load('/content/gdrive/My Drive/models/' + name)
    model.load_state_dict(checkpoint['model'])
    return model

recognizer, _ = load_recognizer()

recognizer

class AudioEnvironment:
    def __init__(self, recognizer, reward_penalty=0, padding=500):
        self.data = None
        self.recognizer = recognizer
        self.recognizer.eval()
        self.audio = None
        self.audio_len = None
        self._left = None
        self.length = None
        self.state = None
        self.reward_penalty = reward_penalty
        self.padding = padding
        self.eval = False
    
    def train(self, data):
        self.data = data
        
    def evaluate(self, audio, policy):
        self.audio = audio
        self.audio_len = len(audio)
        self._left = 0
        self.length = 100
        self.update_state()
        segments = [0]
        total_reward = 0
        while True:
            _, probs = policy(self.state.unsqueeze(0).cuda(), get_action=True)
            action = probs.argmax().item()
            if action == 1:
                segments.append(self._left + self.length)
            _, reward, done = self.step(action)
            total_reward += reward
            if done:
                break
        segments.append(self.audio_len - 1)
        return segments, total_reward
            
    
    def step(self, action):
        reward = 0
        done = False
        # expand window
        if action == 0:
            self.length += 100
            if self._left + self.length + self.padding >= self.audio_len - 1:
                done = True
        # match as phoneme
        elif action == 1:
            reward = self.get_reward() - self.reward_penalty
            self._left += self.length
            self.length = 500
            if self._left + self.length + self.padding >= self.audio_len - 1:
                done = True
        self.update_state()
        return self.state.cpu(), reward, done
        
    def reset(self):
        self.audio = np.random.choice(self.data)
        self.audio_len = len(self.audio)
        self._left = 0
        self.length = 1000
        self.update_state()
        return self.state.cpu()
    
    def update_state(self):
        left = np.max([self._left - self.padding, 0])
        right = np.min([self._left + self.length + self.padding, self.audio_len - 1])
        segment = torch.FloatTensor(self.audio[left:right]).view(1, 1, -1)
        _len = torch.tensor([right - left])
        x, l = segment.cuda(), _len.cuda()
        
        x = F.relu(self.recognizer.conv1(x))
        x = F.relu(self.recognizer.conv1_1(x))
        x = F.relu(self.recognizer.conv2(x))
        x = F.relu(self.recognizer.conv2_1(x))
        x = F.relu(self.recognizer.conv3(x))
        x = F.relu(self.recognizer.conv3_1(x))
        factor = l.max().item() / x.size(2)
        new_lengths = l.type(dtype=torch.float) / factor
        new_lengths = new_lengths.type(dtype=torch.int)
        new_lengths[new_lengths==0] = 1
        x = pack_padded_sequence(x.permute(2, 0, 1), new_lengths)
        out, (h_n, c_n) = self.recognizer.lstm(x)
        h_n = h_n.permute(1, 0, 2).reshape(-1, 1, 3*256)
        self.state = h_n[0,0].detach()
    
    def get_reward(self):
        y = F.relu(self.recognizer.linear(self.state.view(1, 1, -1)))
        y = self.recognizer.linear2(y).squeeze(1)
        return y.item()

env = AudioEnvironment(recognizer)
env.train(audios)
env.reset()
env.step(0)
env.step(1)

"""# Segmentation policy / value network"""

state_dim = 256 * 3
action_dim = 2
class SegmentationPolicyNetwork(nn.Module):
    def __init__(self):
        super(SegmentationPolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input, get_action=True):
        batch_dim, state_dim = input.shape
        input = torch.tensor(input).double()
        action_scores = self.net(input.float())
        action_probs = self.softmax(action_scores)
        if not get_action:
            return action_probs
        # Sample from the distribution if get_action=True
        actions = [np.random.choice(np.arange(action_dim),
                                    p=prob.detach().cpu().numpy())
                   for prob in action_probs]
        return np.array(actions), action_probs

class SegmentationValueNetwork(nn.Module):
    def __init__(self):
        super(SegmentationValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input):
        input = torch.tensor(input).double()
        value = self.net(input.float())
        return value

def generate_rollout(env, policy):
    state = env.reset()
    experience = []
    while True:
        action, prob = policy(torch.tensor(state).cuda().view(1, -1))
        action = action[0]
        prob = prob[0]
        next_state, reward, done = env.step(action)
        experience.append([state.numpy(), reward, action, prob.detach().cpu().numpy(), next_state.numpy()])
        state = next_state
        if done:
            return np.array(experience)

def calculate_returns(rollouts, gamma):
    all_returns = []
    for r in rollouts:
        returns = [0]
        for i, s in enumerate(np.flip(r, axis=0)):
            reward = s[1]
            discounted_sum = gamma*returns[i]
            returns.append(reward + discounted_sum)
        all_returns.append(returns[1:][::-1])
    return all_returns
  
def likelihood(probs, actions):
    return probs[range(probs.shape[0]), actions].unsqueeze(1)

class ExperienceDataset(Dataset):                                                                                                                    
    def __init__(self, experience):                                                                                                                 
        super().__init__()                                                                                                    
        self.exp_joined = []
        for e in experience:
            self.exp_joined.extend(e.tolist())

    def __getitem__(self, index):
        return self.exp_joined[index]

    def __len__(self):                                                                                                                           
        return len(self.exp_joined)                                                                                                                  

def main():

    env = AudioEnvironment(recognizer, reward_penalty=0)
    env.train(audios)
    policy = SegmentationPolicyNetwork()
    value = SegmentationValueNetwork()
    policy.cuda(), value.cuda()

    policy_optim = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.01)
    value_optim = optim.Adam(value.parameters(), lr=1e-3, weight_decay=1)
    
    value_criterion = nn.MSELoss()

    # Hyperparameters
    epochs = 20
    env_samples = 20
    gamma = 0.9
    value_epochs = 2
    policy_epochs = 5
    batch_size = 32
    policy_batch_size = 256
    epsilon = 0.2
    loop = tqdm(total=epochs, position=0, leave=False)
    
    policy_loss = torch.tensor([np.nan])
    avl_list = []
    apl_list = []
    ast_list = []
    for _ in range(epochs):
        # generate rollouts
        rollouts = []
        for _ in range(env_samples):
            # don't forget to reset the environment at the beginning of each episode!
            # rollout for a certain number of steps!
            experience = generate_rollout(env, policy)
            rollouts.append(experience)
            
        
        returns = calculate_returns(rollouts, gamma)

        for i in range(env_samples):
            rollouts[i] = np.column_stack([rollouts[i], returns[i]])

        # Approximate the value function
        dataset = ExperienceDataset(rollouts)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        avl = 0
        for p in policy.parameters():
            p.requires_grad = False
        for p in value.parameters():
            p.requires_grad = True
        for _ in range(value_epochs):
            # train value network
            for states, rewards, actions, probs, next_states, returns in data_loader:
                size = len(returns)
                states = states.cuda()
                value_optim.zero_grad()
                baseline = value(states)
                returns = returns.float().cuda()
                value_loss = value_criterion(baseline, returns.reshape(size, 1))
                value_loss.backward()
                value_optim.step()
                
                avl += value_loss.item()
                loop.set_description('value loss:{:.4f}, policy loss:{:.3f}, returns:{:.3f}'.format(value_loss.item(), policy_loss.item(), returns.mean().item()))
        
        apl = 0
        for p in policy.parameters():
            p.requires_grad = True
        for p in value.parameters():
            p.requires_grad = False
        # Learn a policy
        for _ in range(policy_epochs):
            # train policy network
            for states, rewards, actions, probs, next_states, returns in data_loader:
                size = len(returns)
                states = states.cuda()
                policy_optim.zero_grad()
                baseline = value(states)
                baseline = baseline.detach()

                returns = returns.float().reshape(size, 1).cuda()
                new_probs = policy(states, False)
                
                old_likelihood = likelihood(probs.cuda(), actions)
                new_likelihood = likelihood(new_probs, actions)

                ratio = new_likelihood / old_likelihood
                advantage = returns - baseline
                
                l_1 = ratio * advantage
                l_2 = torch.clamp(ratio, 1-epsilon, 1+epsilon)*advantage
                policy_loss = -torch.mean(torch.min(l_1, l_2))
                policy_loss.backward()
                policy_optim.step()
                
                apl += policy_loss.item()
                loop.set_description('value loss:{:.4f}, policy loss:{:.3f}, returns:{:.3f}'.format(value_loss.item(), policy_loss.item(), returns.mean().item()))
        
        avl /= (len(data_loader) * value_epochs)
        apl /= (len(data_loader) * policy_epochs)
        
        avl_list.append(avl)
        apl_list.append(apl)
                
        loop.update(1)
    
    return avl_list, apl_list, policy, value
                
avl_list, apl_list, policy, value = main()

policy = SegmentationPolicyNetwork()
policy.cuda()
checkpoint = torch.load('/content/gdrive/My Drive/models/policy.pt')
policy.load_state_dict(checkpoint['model'])

env = AudioEnvironment(recognizer)
_segments, reward = env.evaluate(audios_test[0], policy)

def score(audios, segments, policy, recognizer):
    rate = 16000
    env = AudioEnvironment(recognizer)
    dist_list = []
    precision_list = []
    recall_list = []
    for audio, segment in zip(audios, segments):
        pred, _ = env.evaluate(audio, policy)
        target = [int(s) for s in segment.split()[::3]]
        target.append(int(segment.split()[-2]))
        target = np.array(target)
        pred = np.vstack([pred, np.zeros_like(pred)]).T
        target = np.vstack([target, np.ones_like(target)]).T
        combined = np.stack(sorted(np.vstack([pred, target]), key=lambda x: x[0]))
        pairs = []
        for i,c in enumerate(combined):
            if c[1] == 0:
                before = [None, None]
                after = [None, None]
                for cj in combined[i::-1]:
                    if cj[1] == 1:
                        before = cj
                        break
                for cj in combined[i:]:
                    if cj[1] == 1:
                        after = cj
                        break
                if before[0] != None and after[0] != None:
                    if abs(c[0] - before[0]) >= abs(c[0] - after[0]):
                        pair = np.array([c[0], after[0], abs(c[0] - after[0])])
                    else:
                        pair = np.array([c[0], before[0], abs(c[0] - before[0])])
                        
                elif before[0] == None:
                    pair = np.array([c[0], after[0], abs(c[0] - after[0])])
                    
                elif after[0] == None:
                    pair = np.array([c[0], before[0], abs(c[0] - before[0])])
                pairs.append(pair)
        pairs = np.array(pairs)
        best_pairs = []
        for u in np.unique(pairs[:, 1]):
            best = sorted(pairs[pairs[:, 1] == u], key=lambda x: x[2])[0]
            best_pairs.append(best)
        best_pairs = np.array(best_pairs)
        l, _ = best_pairs.shape
        dist = np.mean(best_pairs, axis=0)[-1] / rate
        precision = l / len(pred)
        recall = l / len(target)
        dist_list.append(dist)
        precision_list.append(precision)
        recall_list.append(recall)
    return np.mean(dist_list), np.mean(precision_list), np.mean(recall_list)

score(audios_test, segments_test, policy, recognizer)

classifier = load_classifier()

s0 = _segments[0]
phones = np.array([
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em',
    'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm',
    'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w',
    'y', 'z', 'zh'
])
phone_to_idx = {
    phone: i for i, phone in enumerate(phones)
}
for s in _segments[1:]:
    sample = torch.FloatTensor(audios_test[0][max(s0 - 500, 0):s + 500])
    length = torch.tensor(len(sample))
    sample, length = sample.view(1, 1, -1).cuda(), length.unsqueeze(0).cuda()
    r = recognizer(sample, length)
    p = classifier(sample, length)
    idx = p.topk(3)[1].cpu()
    print(s0, s, phones[idx], r.item())
    s0 = s

plt.plot(audios[0])

print(segments_test[0])

