"""
Proximal Policy Optimization (PPO) Agent for VANET Environment
Implements PPO with Actor-Critic architecture
"""

import zmq
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import os
from datetime import datetime


class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128]):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
    def act(self, state):
        """Select action based on policy"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, state_value
    
    def evaluate(self, states, actions):
        """Evaluate actions for PPO update"""
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy


class PPOMemory:
    """Memory buffer for PPO experiences"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPOAgent:
    """PPO Agent for VANET parameter optimization"""
    
    def __init__(self, state_dim=9, learning_rate=0.0003, gamma=0.95,
                 eps_clip=0.2, K_epochs=4, entropy_coef=0.01, value_coef=0.5):
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
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
        print(f"[PPO] Using device: {self.device}")
        
        # Policy network
        self.policy = ActorCritic(state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Old policy for PPO ratio
        self.policy_old = ActorCritic(state_dim, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Loss function
        self.MseLoss = nn.MSELoss()
        
        # Memory
        self.memory = PPOMemory()
        
        # Statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.current_episode_reward = 0
        self.current_episode_losses = []
    
    def normalize_state(self, state_dict):
        """Normalize state features to similar scales"""
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
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                action_idx, action_logprob, state_value = self.policy_old.act(state_tensor)
                # Store in memory for PPO update
                self.memory.states.append(state)
                self.memory.actions.append(action_idx)
                self.memory.logprobs.append(action_logprob)
                self.memory.state_values.append(state_value)
            else:
                # Greedy selection for evaluation
                action_probs, _ = self.policy(state_tensor)
                action_idx = action_probs.argmax().item()
        
        return action_idx
    
    def store_reward_done(self, reward, done):
        """Store reward and done flag"""
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)
    
    def update(self):
        """PPO update using collected experiences"""
        if len(self.memory.states) == 0:
            return None
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), 
                                       reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert to tensors
        old_states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        old_actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_logprobs = torch.stack(self.memory.logprobs).detach().to(self.device)
        old_state_values = torch.stack(self.memory.state_values).squeeze().detach().to(self.device)
        
        # Calculate advantages
        advantages = rewards - old_state_values
        
        # PPO update for K epochs
        total_loss = 0
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            
            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, rewards)
            entropy_loss = -dist_entropy.mean()
            
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / self.K_epochs
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory.clear()
        
        return avg_loss
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
        }, filepath)
        print(f"[PPO] Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"[PPO] Model loaded from {filepath}")


def run_ppo_agent(port=5555, train=True, episodes=100, max_steps=20,
                  save_interval=10, model_path=None, update_interval=20):
    """Run PPO agent and communicate with NS3 environment"""
    
    # Create agent
    agent = PPOAgent()
    
    # Load model if specified
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"[PPO] Loaded existing model from {model_path}")
    
    # Setup ZMQ
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    print(f"[PPO] Agent listening on port {port}")
    print(f"[PPO] Mode: {'TRAINING' if train else 'EVALUATION'}")
    print(f"[PPO] Episodes: {episodes}, Max steps per episode: {max_steps}")
    print(f"[PPO] Action space: {agent.action_dim} actions")
    print(f"[PPO] Update interval: {update_interval} steps")
    print(f"[PPO] Waiting for NS3 environment...\n")
    
    episode = 0
    step = 0
    total_steps = 0
    
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
            
            print(f"[PPO] Episode {episode+1}/{episodes}, Step {step+1}/{max_steps}, Total: {total_steps}")
            print(f"      Time: {state_dict.get('time', 0):.1f}s, "
                  f"PDR: {state_dict.get('PDR', 0):.3f}, "
                  f"Neighbors: {state_dict.get('avgNeighbors', 0):.1f}, "
                  f"Throughput: {state_dict.get('throughput', 0):.0f}")
            
            # Select action
            action_idx = agent.select_action(current_state, training=train)
            action = agent.actions[action_idx]
            
            print(f"      Action: beaconHz={action['beaconHz']}, txPower={action['txPower']}")
            
            # Send action
            socket.send_string(json.dumps({"action": action}))
            
            # Receive reward
            msg = socket.recv()
            reward_msg = json.loads(msg.decode())
            reward = reward_msg.get('reward', 0.0)
            done = reward_msg.get('done', False)
            
            print(f"      Reward: {reward:.3f}, Done: {done}")
            
            agent.current_episode_reward += reward
            
            # Store reward and done in memory
            if train:
                agent.store_reward_done(reward, done)
            
            # Send acknowledgment
            socket.send_string("ack")
            
            step += 1
            total_steps += 1
            
            # PPO update at intervals
            if train and total_steps % update_interval == 0:
                loss = agent.update()
                if loss is not None:
                    agent.current_episode_losses.append(loss)
                    print(f"      >>> PPO Update - Loss: {loss:.4f} <<<")
            
            # Episode end
            if done or step >= max_steps:
                # Final update for episode
                if train:
                    loss = agent.update()
                    if loss is not None:
                        agent.current_episode_losses.append(loss)
                
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
                    model_file = f"models/ppo_vanet_ep{episode+1}_{timestamp}.pth"
                    agent.save_model(model_file)
                
                # Reset for next episode
                agent.current_episode_reward = 0
                agent.current_episode_losses = []
                step = 0
                episode += 1
                
                print(f"[PPO] Starting episode {episode+1}/{episodes}...\n")
    
    except KeyboardInterrupt:
        print("\n[PPO] Training interrupted by user")
    
    finally:
        # Final save
        if train:
            final_model = f"models/ppo_vanet_final_{timestamp}.pth"
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
    parser = argparse.ArgumentParser(description='PPO Agent for VANET')
    parser.add_argument('--port', type=int, default=5555, help='ZMQ port')
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=20, help='Max steps per episode')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N episodes')
    parser.add_argument('--update_interval', type=int, default=20, help='PPO update every N steps')
    parser.add_argument('--model', type=str, default=None, help='Path to model to load')
    
    args = parser.parse_args()
    
    train_mode = args.train or not args.eval
    
    run_ppo_agent(
        port=args.port,
        train=train_mode,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        model_path=args.model,
        update_interval=args.update_interval
    )
