"""
Simple Continuous DQN Agent for VANET - BeaconHz Only
========================================================

EXPLANATION OF DQN AND EPISODES:
---------------------------------

1. WHAT IS DQN (Deep Q-Network)?
   - DQN is a reinforcement learning algorithm that learns to make decisions
   - It uses a neural network to estimate Q-values (quality of actions)
   - Q-value = expected future reward for taking action A in state S
   - The agent learns which actions lead to better outcomes over time

2. HOW DQN WORKS:
   Step 1: Observe current state (PDR, throughput, neighbors, etc.)
   Step 2: Choose action (change beaconHz) using epsilon-greedy:
           - With probability ε: explore (random action)
           - With probability 1-ε: exploit (best known action)
   Step 3: Execute action in environment (NS3 simulation)
   Step 4: Receive reward based on performance
   Step 5: Store experience (state, action, reward, next_state)
   Step 6: Learn from past experiences using "Experience Replay"
   Step 7: Update Q-value estimates to improve future decisions

3. WHAT ARE EPISODES?
   Traditional RL:
   - Episode = one complete run of a simulation from start to finish
   - Like playing one game of chess or one race in a driving sim
   - Episode ends when reaching terminal state (crash, finish, timeout)
   
   In your case (OLD approach with 10 runs):
   - You run NS3 simulation 10 separate times
   - Each run = 1 episode
   - Agent resets between runs
   - Problem: Inefficient, not real-time adaptive

4. CONTINUOUS LEARNING (NEW approach - what we're doing here):
   - ONE long-running NS3 simulation
   - Agent learns continuously while simulation runs
   - No episodes - just continuous decision making
   - More realistic: adapts in real-time like a real system would
   - Steps = decision points (every time agent gets new state)

5. KEY DQN COMPONENTS:
   a) Policy Network: Neural net that outputs Q-values for each action
   b) Target Network: Stable copy of policy net (updated slowly)
   c) Replay Buffer: Memory of past experiences for learning
   d) Epsilon-greedy: Balance between exploration and exploitation
   e) Bellman Equation: Q(s,a) = reward + γ * max Q(next_state, next_action)

6. THIS IMPLEMENTATION (Continuous, BeaconHz only):
   - No episodes! Just continuous learning
   - Only controls beacon frequency (2-12 Hz)
   - Keeps txPower fixed at 23 dBm
   - Learns from every state transition
   - Adapts in real-time to changing network conditions
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

# Optional WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] WandB not installed. Run 'pip install wandb' for experiment tracking.")

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Simple Deep Q-Network for beacon frequency control
    Input: 9 state features (includes CBR, removed txPower since we don't control it)
    Output: 6 Q-values (one for each beacon frequency option)
    """
    
    def __init__(self, state_dim=9, action_dim=6, hidden_size=64):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer
    Stores past experiences and samples random batches for training
    This breaks temporal correlations and improves learning stability
    """
    
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        """Add new experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class ContinuousDQNAgent:
    """
    Continuous Learning DQN Agent for VANET BeaconHz Control
    
    This agent:
    - Learns continuously (no episodes)
    - Only controls beacon frequency (beaconHz)
    - Keeps txPower fixed at 23 dBm
    - Updates neural network after each step
    - Uses experience replay for stable learning
    """
    
    def __init__(self, 
                 state_dim=9,           # 9 state features (includes CBR, no txPower)
                 learning_rate=0.0005,   # How fast to learn
                 gamma=0.99,            # Discount factor (how much to value future rewards)
                 epsilon_start=1.0,     # Initial exploration rate
                 epsilon_end=0.05,       # Minimum exploration rate
                 epsilon_decay=0.999,  # Decay rate per step
                 buffer_size=10000,      # Replay buffer capacity
                 batch_size=64,         # Training batch size
                 target_update_freq=200): # Update target network every N steps
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0
        
        # SIMPLIFIED ACTION SPACE: Only beacon frequency
        # BeaconHz options: [2, 4, 6, 8, 10, 12] Hz
        self.beacon_options = [2, 4, 6, 8, 10, 12]
        self.action_dim = len(self.beacon_options)
        
        # Fixed txPower
        self.fixed_tx_power = 23  # dBm
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Using device: {self.device}")
        
        # Create policy and target networks
        self.policy_net = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Optimizer and loss function
        self.learning_rate = learning_rate  # Store for later reference
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Statistics tracking
        self.total_reward = 0
        self.recent_rewards = deque(maxlen=100)  # Track last 100 rewards
        self.recent_losses = deque(maxlen=100)   # Track last 100 losses
    
    def normalize_state(self, state_dict):
        """
        Normalize state features to 0-1 range for better neural network performance
        
        State features (9 total):
        1. PDR (Packet Delivery Ratio) - already 0-1
        2. avgNeighbors - normalized by assumed max of 20
        3. beaconHz - normalized by max 20 Hz
        4. numVehicles - normalized by assumed max of 100
        5. packetsReceived - normalized by typical max
        6. packetsSent - normalized by typical max
        7. throughput - normalized by typical max
        8. time - normalized by simulation time
        9. CBR (Channel Busy Ratio) - already 0-1
        
        Note: We removed txPower from state since we don't control it
        """
        normalized = np.array([
            state_dict.get('PDR', 0.0),                          # 0-1 range
            state_dict.get('avgNeighbors', 0.0) / 20.0,         # 0-1 range
            state_dict.get('beaconHz', 8.0) / 20.0,             # 0-1 range
            state_dict.get('numVehicles', 10.0) / 100.0,        # 0-1 range
            state_dict.get('packetsReceived', 0.0) / 10000.0,   # 0-1 range
            state_dict.get('packetsSent', 0.0) / 1000.0,        # 0-1 range
            state_dict.get('throughput', 0.0) / 100000.0,       # 0-1 range
            state_dict.get('time', 0.0) / 200.0,                # 0-1 range (assuming 200s max)
            state_dict.get('CBR', 0.0),                         # 0-1 range (already normalized)
        ], dtype=np.float32)
        
        return normalized
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection
        
        With probability epsilon: choose random action (EXPLORE)
        With probability 1-epsilon: choose best action based on Q-values (EXPLOIT)
        
        Returns: action_idx (0-5 corresponding to beacon frequencies 2-12 Hz)
        """
        if training and random.random() < self.epsilon:
            # EXPLORATION: Random action
            action_idx = random.randrange(self.action_dim)
        else:
            # EXPLOITATION: Best action according to policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
        
        return action_idx
    
    def get_action_params(self, action_idx):
        """
        Convert action index to actual parameters
        Returns: dict with beaconHz and txPower
        """
        return {
            'beaconHz': self.beacon_options[action_idx],
            'txPower': self.fixed_tx_power  # Always 23 dBm
        }
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action_idx, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def train_step(self):
        """
        Perform one training step using experience replay
        
        This is where the actual learning happens:
        1. Sample random batch from replay buffer
        2. Compute current Q-values from policy network
        3. Compute target Q-values using Bellman equation
        4. Update policy network to minimize difference
        5. Periodically update target network
        
        Returns: loss value or None if not enough experiences
        """
        # Need enough experiences before we can train
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample random batch of experiences
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch into separate tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1).to(self.device)
        
        # Compute current Q-values: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute target Q-values using Bellman equation:
        # Q_target(s, a) = reward + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (Mean Squared Error)
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically (for stability)
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[DQN] Target network updated at step {self.step_counter}")
        
        return loss.item()
    
    def decay_epsilon(self):
        """Gradually decrease exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_counter': self.step_counter,
            'total_reward': self.total_reward,
        }, filepath)
        print(f"[DQN] Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.step_counter = checkpoint.get('step_counter', 0)
        self.total_reward = checkpoint.get('total_reward', 0)
        print(f"[DQN] Model loaded from {filepath}")


def run_continuous_dqn(port=5555, train=True, max_steps=1000, 
                       save_interval=100, model_path=None, use_wandb=False):
    """
    Run DQN agent in continuous learning mode
    
    NO EPISODES - just continuous learning!
    
    Args:
        port: ZMQ port for communication with NS3
        train: If True, agent learns. If False, agent only exploits
        max_steps: Maximum number of steps (0 = unlimited)
        save_interval: Save model every N steps
        model_path: Path to existing model to load
        use_wandb: If True, log to Weights & Biases
    """
    
    # Create agent
    agent = ContinuousDQNAgent()
    
    # Load existing model if specified
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
    
    # Setup ZMQ communication with NS3
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    # Initialize WandB if requested and available
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE and train:
        try:
            wandb_run = wandb.init(
                project="vanet-dqn-with-ns3",
                name=f"dqn_continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "learning_rate": agent.learning_rate,
                    "gamma": agent.gamma,
                    "epsilon_start": agent.epsilon,
                    "epsilon_end": agent.epsilon_end,
                    "epsilon_decay": agent.epsilon_decay,
                    "batch_size": agent.batch_size,
                    "buffer_size": agent.replay_buffer.buffer.maxlen,
                    "target_update_freq": agent.target_update_freq,
                    "fixed_tx_power": agent.fixed_tx_power,
                    "state_dim": agent.state_dim,
                    "action_dim": agent.action_dim,
                    "beacon_options": agent.beacon_options,
                    "max_steps": max_steps,
                    "save_interval": save_interval,
                },
                tags=["dqn", "vanet", "beaconhz-only", "continuous-learning"]
            )
            print("[INFO] WandB logging enabled")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")
            use_wandb = False
    elif use_wandb and not WANDB_AVAILABLE:
        print("[WARNING] WandB requested but not installed. Run: pip install wandb")
        use_wandb = False

    print("\n" + "="*70)
    print("CONTINUOUS DQN AGENT FOR VANET - BeaconHz Control Only")
    print("="*70)
    print(f"Mode: {'TRAINING (Learning continuously)' if train else 'EVALUATION (No learning)'}")
    print(f"Port: {port}")
    print(f"Action Space: {agent.action_dim} actions (BeaconHz: {agent.beacon_options})")
    print(f"TxPower: FIXED at {agent.fixed_tx_power} dBm")
    print(f"Max Steps: {'Unlimited' if max_steps == 0 else max_steps}")
    print(f"Current Epsilon: {agent.epsilon:.4f}")
    print(f"Device: {agent.device}")
    print("="*70)
    print("\nWaiting for NS3 simulation to connect...\n")
    
    step = 0
    prev_state = None
    prev_action_idx = None
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        while max_steps == 0 or step < max_steps:
            # Receive state from NS3
            msg = socket.recv()
            data = json.loads(msg.decode())
            
            # Handle non-state messages
            if data.get('type') != 'state':
                socket.send_string(json.dumps({
                    "action": {
                        "beaconHz": 8, 
                        "txPower": agent.fixed_tx_power
                    }
                }))
                continue
            
            # Extract and normalize state
            state_dict = data.get('data', {})
            current_state = agent.normalize_state(state_dict)
            
            # Display current status
            if step % 10 == 0 or step < 5:  # Print every 10 steps or first 5
                print(f"\n{'='*70}")
                print(f"Step {step + 1}")
                print(f"{'='*70}")
            
            print(f"[State] Time: {state_dict.get('time', 0):.1f}s | "
                  f"PDR: {state_dict.get('PDR', 0):.3f} | "
                  f"Neighbors: {state_dict.get('avgNeighbors', 0):.1f} | "
                  f"Throughput: {state_dict.get('throughput', 0):.0f} bps | "
                  f"CBR: {state_dict.get('CBR', 0):.3f}")
            
            # Select action
            action_idx = agent.select_action(current_state, training=train)
            action = agent.get_action_params(action_idx)
            
            print(f"[Action] BeaconHz: {action['beaconHz']} Hz | "
                  f"TxPower: {action['txPower']} dBm (FIXED)")
            if train:
                print(f"[Learning] Epsilon: {agent.epsilon:.4f} | "
                      f"Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.buffer.maxlen}")
            
            # Send action to NS3
            socket.send_string(json.dumps({"action": action}))
            
            # Receive reward from NS3
            msg = socket.recv()
            reward_msg = json.loads(msg.decode())
            reward = reward_msg.get('reward', 0.0)
            done = reward_msg.get('done', False)
            
            print(f"[Feedback] Reward: {reward:.3f} | Done: {done}")
            
            # Update statistics
            agent.total_reward += reward
            agent.recent_rewards.append(reward)
            
            # Learn from experience (if not first step)
            loss = None  # Initialize loss
            if prev_state is not None and train:
                # Store experience
                agent.store_experience(prev_state, prev_action_idx, reward, 
                                      current_state, done)
                
                # Train on batch
                loss = agent.train_step()
                if loss is not None:
                    agent.recent_losses.append(loss)
                    print(f"[Training] Loss: {loss:.4f}")
                
                # Decay exploration rate
                agent.decay_epsilon()
            
            # Log to WandB if enabled
            if use_wandb and wandb_run:
                log_data = {
                    "step": step,
                    "reward": reward,
                    "total_reward": agent.total_reward,
                    "epsilon": agent.epsilon,
                    "buffer_size": len(agent.replay_buffer),
                    "chosen_beaconHz": action['beaconHz'],
                    "state_pdr": state_dict.get('PDR', 0),
                    "state_avg_neighbors": state_dict.get('avgNeighbors', 0),
                    "state_throughput": state_dict.get('throughput', 0),
                    "state_cbr": state_dict.get('CBR', 0),
                }
                if loss is not None:
                    log_data["loss"] = loss
                
                wandb.log(log_data)

            # Print summary statistics periodically
            if step > 0 and step % 50 == 0:
                avg_reward = np.mean(agent.recent_rewards) if agent.recent_rewards else 0
                avg_loss = np.mean(agent.recent_losses) if agent.recent_losses else 0
                print(f"\n{'*'*70}")
                print(f"PROGRESS SUMMARY (Last 50 steps)")
                print(f"{'*'*70}")
                print(f"Total Steps: {step}")
                print(f"Total Reward: {agent.total_reward:.2f}")
                print(f"Avg Recent Reward: {avg_reward:.3f}")
                if train and agent.recent_losses:
                    print(f"Avg Recent Loss: {avg_loss:.4f}")
                print(f"Exploration Rate (Epsilon): {agent.epsilon:.4f}")
                print(f"{'*'*70}\n")
            
            # Send acknowledgment
            socket.send_string("ack")
            
            # Save model periodically
            if train and save_interval > 0 and (step + 1) % save_interval == 0:
                model_file = f"models/dqn_continuous_step{step+1}_{timestamp}.pth"
                agent.save_model(model_file)
            
            # Update for next iteration
            prev_state = current_state
            prev_action_idx = action_idx
            step += 1
            
            # Check if simulation ended
            if done:
                print("\n[INFO] Simulation ended (done=True received)")
                break
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user (Ctrl+C)")
    
    finally:
        # Final save
        if train:
            final_model = f"models/dqn_continuous_final_{timestamp}.pth"
            agent.save_model(final_model)
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Total Steps: {step}")
        print(f"Total Reward: {agent.total_reward:.2f}")
        if agent.recent_rewards:
            print(f"Average Recent Reward (last 100): {np.mean(agent.recent_rewards):.3f}")
            print(f"Max Recent Reward: {np.max(agent.recent_rewards):.3f}")
            print(f"Min Recent Reward: {np.min(agent.recent_rewards):.3f}")
        if train and agent.recent_losses:
            print(f"Average Recent Loss (last 100): {np.mean(agent.recent_losses):.4f}")
        print(f"Final Epsilon: {agent.epsilon:.4f}")
        print(f"Replay Buffer Size: {len(agent.replay_buffer)}")
        print("="*70)
        
        # Close WandB if it was initialized
        if use_wandb and wandb_run:
            wandb.finish()
            print("[INFO] WandB run finished")
        
        socket.close()
        ctx.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Continuous DQN Agent for VANET (BeaconHz control only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training mode (continuous learning)
  python dqn_simple_continuous.py --train --max_steps 500
  
  # Evaluation mode (no learning, uses learned policy)
  python dqn_simple_continuous.py --eval --model models/dqn_continuous_final_XXX.pth
  
  # Unlimited continuous learning (until Ctrl+C)
  python dqn_simple_continuous.py --train --max_steps 0
        """
    )
    
    parser.add_argument('--port', type=int, default=5555, 
                       help='ZMQ port for NS3 communication (default: 5555)')
    parser.add_argument('--train', action='store_true', 
                       help='Enable training mode (agent learns continuously)')
    parser.add_argument('--eval', action='store_true', 
                       help='Enable evaluation mode (no learning, exploitation only)')
    parser.add_argument('--max_steps', type=int, default=0, 
                       help='Maximum steps (0=unlimited, default: 0)')
    parser.add_argument('--save_interval', type=int, default=100, 
                       help='Save model every N steps (default: 100)')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to model file to load')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging (requires: pip install wandb)')
    
    args = parser.parse_args()
    
    # Default to training mode if neither specified
    train_mode = args.train or not args.eval
    
    run_continuous_dqn(
        port=args.port,
        train=train_mode,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        model_path=args.model,
        use_wandb=args.wandb
    )
