"""
Deep Q-Network (DQN) Agent for VANET Environment
Implements DQN with Experience Replay and Target Network
"""

import zmq
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import argparse
import os
from datetime import datetime

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """Deep Q-Network with 3 hidden layers"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for VANET parameter optimization"""
    
    def __init__(self, state_dim=9, learning_rate=0.001, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=10):
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Define discrete action space
        # Actions: (beaconHz, txPower) combinations
        # beaconHz: [2, 4, 6, 8, 10, 12] Hz
        # txPower: [20, 23, 26, 30] dBm
        self.beacon_options = [2, 4, 6, 8, 10, 12]
        self.power_options = [20, 23, 26, 30]
        
        # Create all combinations
        self.actions = []
        for beacon in self.beacon_options:
            for power in self.power_options:
                self.actions.append({'beaconHz': beacon, 'txPower': power})
        
        self.action_dim = len(self.actions)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Using device: {self.device}")
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.current_episode_reward = 0
        self.current_episode_losses = []
    
    def normalize_state(self, state_dict):
        """Normalize state features to similar scales"""
        # Extract and normalize features
        normalized = np.array([
            state_dict.get('PDR', 0.0),  # Already 0-1
            state_dict.get('avgNeighbors', 0.0) / 20.0,  # Assume max 20 neighbors
            state_dict.get('beaconHz', 8.0) / 20.0,  # Max 20 Hz
            state_dict.get('numVehicles', 10.0) / 100.0,  # Assume max 100 vehicles
            state_dict.get('packetsReceived', 0.0) / 10000.0,  # Normalize by typical max
            state_dict.get('packetsSent', 0.0) / 1000.0,  # Normalize by typical max
            state_dict.get('throughput', 0.0) / 100000.0,  # Max ~100 kbps
            state_dict.get('time', 0.0) / 100.0,  # Normalize by sim time
            state_dict.get('txPower', 23.0) / 30.0,  # Max 30 dBm
        ], dtype=np.float32)
        
        return normalized
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            # Random action
            action_idx = random.randrange(self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
        
        return action_idx
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action_idx, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
        }, filepath)
        print(f"[DQN] Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"[DQN] Model loaded from {filepath}")


def run_dqn_agent(port=5555, train=True, episodes=100, max_steps=20, 
                  save_interval=10, model_path=None):
    """Run DQN agent and communicate with NS3 environment"""
    
    # Create agent
    agent = DQNAgent()
    
    # Load model if specified
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"[DQN] Loaded existing model from {model_path}")
    
    # Setup ZMQ
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    print(f"[DQN] Agent listening on port {port}")
    print(f"[DQN] Mode: {'TRAINING' if train else 'EVALUATION'}")
    print(f"[DQN] Episodes: {episodes}, Max steps per episode: {max_steps}")
    print(f"[DQN] Action space: {agent.action_dim} actions")
    print(f"[DQN] Waiting for NS3 environment...\n")
    
    episode = 0
    step = 0
    prev_state = None
    prev_action_idx = None
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        while episode < episodes:
            # Receive state
            msg = socket.recv()
            data = json.loads(msg.decode())
            
            if data.get('type') != 'state':
                socket.send_string(json.dumps({"action": {"beaconHz": 8, "txPower": 23}}))
                continue
            
            state_dict = data.get('data', {})
            current_state = agent.normalize_state(state_dict)
            
            print(f"[DQN] Episode {episode+1}/{episodes}, Step {step+1}/{max_steps}")
            print(f"       Time: {state_dict.get('time', 0):.1f}s, "
                  f"PDR: {state_dict.get('PDR', 0):.3f}, "
                  f"Neighbors: {state_dict.get('avgNeighbors', 0):.1f}, "
                  f"Throughput: {state_dict.get('throughput', 0):.0f}")
            
            # Select action
            action_idx = agent.select_action(current_state, training=train)
            action = agent.actions[action_idx]
            
            print(f"       Action: beaconHz={action['beaconHz']}, txPower={action['txPower']}")
            if train:
                print(f"       Epsilon: {agent.epsilon:.4f}")
            
            # Send action
            socket.send_string(json.dumps({"action": action}))
            
            # Receive reward
            msg = socket.recv()
            reward_msg = json.loads(msg.decode())
            reward = reward_msg.get('reward', 0.0)
            done = reward_msg.get('done', False)
            
            print(f"       Reward: {reward:.3f}, Done: {done}")
            
            agent.current_episode_reward += reward
            
            # Store experience and train (if not first step)
            if prev_state is not None and train:
                agent.store_experience(prev_state, prev_action_idx, reward, 
                                       current_state, done)
                loss = agent.train_step()
                if loss is not None:
                    agent.current_episode_losses.append(loss)
                    print(f"       Loss: {loss:.4f}, Buffer size: {len(agent.replay_buffer)}")
            
            # Send acknowledgment
            socket.send_string("ack")
            
            # Update for next iteration
            prev_state = current_state
            prev_action_idx = action_idx
            step += 1
            
            # Episode end
            if done or step >= max_steps:
                print(f"\n{'='*60}")
                print(f"Episode {episode+1} finished!")
                print(f"Total Reward: {agent.current_episode_reward:.3f}")
                if agent.current_episode_losses:
                    avg_loss = np.mean(agent.current_episode_losses)
                    print(f"Average Loss: {avg_loss:.4f}")
                    agent.episode_losses.append(avg_loss)
                print(f"{'='*60}\n")
                
                agent.episode_rewards.append(agent.current_episode_reward)
                
                # Save model periodically
                if train and (episode + 1) % save_interval == 0:
                    model_file = f"models/dqn_vanet_ep{episode+1}_{timestamp}.pth"
                    agent.save_model(model_file)
                
                # Reset for next episode
                agent.current_episode_reward = 0
                agent.current_episode_losses = []
                prev_state = None
                prev_action_idx = None
                step = 0
                episode += 1
                
                if train:
                    agent.decay_epsilon()
                
                print(f"[DQN] Starting episode {episode+1}/{episodes}...\n")
    
    except KeyboardInterrupt:
        print("\n[DQN] Training interrupted by user")
    
    finally:
        # Final save
        if train:
            final_model = f"models/dqn_vanet_final_{timestamp}.pth"
            agent.save_model(final_model)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY" if train else "EVALUATION SUMMARY")
        print("="*60)
        print(f"Episodes completed: {len(agent.episode_rewards)}")
        if agent.episode_rewards:
            print(f"Average reward: {np.mean(agent.episode_rewards):.3f}")
            print(f"Max reward: {np.max(agent.episode_rewards):.3f}")
            print(f"Min reward: {np.min(agent.episode_rewards):.3f}")
        if agent.episode_losses:
            print(f"Average loss: {np.mean(agent.episode_losses):.4f}")
        print("="*60)
        
        socket.close()
        ctx.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Agent for VANET')
    parser.add_argument('--port', type=int, default=5555, help='ZMQ port')
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=20, help='Max steps per episode')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N episodes')
    parser.add_argument('--model', type=str, default=None, help='Path to model to load')
    
    args = parser.parse_args()
    
    train_mode = args.train or not args.eval
    
    run_dqn_agent(
        port=args.port,
        train=train_mode,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        model_path=args.model
    )
