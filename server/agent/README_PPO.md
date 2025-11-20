# PPO Agent for VANET Scenario

This directory contains Proximal Policy Optimization (PPO) agents for the VANET (Vehicular Ad-hoc Network) scenario. The PPO agents use an Actor-Critic architecture to learn optimal policies for controlling vehicle communication parameters.

## Available PPO Agents

### 1. **ppo_dual_continuous.py** (Recommended)
Continuous learning PPO agent that controls **both BeaconHz and TxPower** simultaneously.

**Features:**
- **Dual Control**: Optimizes both beacon frequency (2-12 Hz) and transmission power (15-30 dBm)
- **36 Discrete Actions**: 6 BeaconHz × 6 TxPower combinations
- **Continuous Learning**: No episodes, learns continuously from experience
- **Actor-Critic Architecture**: Separate policy (actor) and value (critic) networks
- **PPO Clipping**: Stable policy updates with importance sampling ratio clipping
- **WandB Integration**: Optional experiment tracking
- **10 State Features**: Including PDR, neighbors, throughput, CBR, etc.

**Action Space:**
- BeaconHz: [2, 4, 6, 8, 10, 12] Hz
- TxPower: [15, 18, 21, 23, 26, 30] dBm
- Total: 36 action combinations

### 2. **ppo_agent.py**
Episodic PPO agent (older version).

**Features:**
- Episode-based learning
- Simpler implementation
- Fewer action options

## PPO vs DQN

### Proximal Policy Optimization (PPO)
**Pros:**
- ✅ On-policy learning (more stable)
- ✅ Better for continuous action spaces (can be adapted)
- ✅ More sample efficient in some scenarios
- ✅ Explicit exploration through policy entropy
- ✅ Monotonic improvement guarantees (with clipped objective)

**Cons:**
- ❌ Requires more memory (stores full trajectories)
- ❌ Can be slower to train (multiple epochs per update)
- ❌ More hyperparameters to tune

### Deep Q-Network (DQN)
**Pros:**
- ✅ Off-policy learning (sample efficient)
- ✅ Experience replay (learns from past experiences)
- ✅ Simpler implementation
- ✅ Works well with discrete actions

**Cons:**
- ❌ Can be unstable without proper tuning
- ❌ Epsilon-greedy exploration can be suboptimal
- ❌ Requires target network updates

## Usage

### Training Mode (Continuous Learning)

```bash
# Basic training (unlimited steps)
python ppo_dual_continuous.py --train --max_steps 0

# Training with limited steps
python ppo_dual_continuous.py --train --max_steps 1000

# Training with WandB logging
python ppo_dual_continuous.py --train --max_steps 1000 --wandb

# Training with custom update interval
python ppo_dual_continuous.py --train --max_steps 1000 --update_interval 256

# Training with frequent model saving
python ppo_dual_continuous.py --train --max_steps 1000 --save_interval 50
```

### Evaluation Mode (No Learning)

```bash
# Evaluate a trained model
python ppo_dual_continuous.py --eval --model models/ppo_dual_final_20251111_120000.pth

# Evaluate for specific number of steps
python ppo_dual_continuous.py --eval --model models/ppo_dual_final_XXX.pth --max_steps 500
```

### Advanced Options

```bash
# Custom ZMQ port
python ppo_dual_continuous.py --train --port 5556

# Resume training from checkpoint
python ppo_dual_continuous.py --train --model models/ppo_dual_step500_XXX.pth
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--port` | int | 5555 | ZMQ port for NS3 communication |
| `--train` | flag | - | Enable training mode |
| `--eval` | flag | - | Enable evaluation mode (no learning) |
| `--max_steps` | int | 0 | Maximum steps (0=unlimited) |
| `--save_interval` | int | 100 | Save model every N steps |
| `--model` | str | None | Path to model file to load |
| `--wandb` | flag | - | Enable Weights & Biases logging |
| `--update_interval` | int | 128 | PPO update interval in timesteps |

## State Features (10 dimensions)

The agent observes 10 normalized state features:

1. **PDR** (Packet Delivery Ratio): 0-1 range
2. **avgNeighbors**: Average number of neighboring vehicles
3. **beaconHz**: Current beacon frequency
4. **txPower**: Current transmission power
5. **numVehicles**: Total number of vehicles
6. **packetsReceived**: Total packets received
7. **packetsSent**: Total packets sent
8. **throughput**: Current throughput (bps)
9. **time**: Simulation time
10. **CBR** (Channel Busy Ratio): 0-1 range

## Reward Function

The reward is designed to optimize:
- **High PDR** (Packet Delivery Ratio)
- **Good connectivity** (appropriate number of neighbors)
- **Low CBR** (Channel Busy Ratio - avoid congestion)
- **Energy efficiency** (lower transmission power when appropriate)

## PPO Hyperparameters

Default hyperparameters in `ppo_dual_continuous.py`:

```python
learning_rate = 0.0003       # Adam learning rate
gamma = 0.99                 # Discount factor
eps_clip = 0.2               # PPO clipping parameter
K_epochs = 4                 # Number of PPO update epochs
entropy_coef = 0.01          # Entropy coefficient (exploration)
value_coef = 0.5             # Value loss coefficient
update_timesteps = 128       # Update policy every N timesteps
```

## Model Files

Trained models are saved in the `models/` directory with timestamps:

- **Periodic checkpoints**: `ppo_dual_step{N}_{timestamp}.pth`
- **Final model**: `ppo_dual_final_{timestamp}.pth`

Each checkpoint contains:
- Policy network weights
- Old policy network weights
- Optimizer state
- Training statistics
- Action mapping

## Architecture Details

### Actor-Critic Network

**Shared Layers:**
- Linear(10, 128) + ReLU + Dropout(0.2)
- Linear(128, 128) + ReLU + Dropout(0.2)

**Actor Head (Policy):**
- Linear(128, 64) + ReLU
- Linear(64, 36) → Action logits

**Critic Head (Value):**
- Linear(128, 64) + ReLU
- Linear(64, 1) → State value

**Total Parameters**: ~25K parameters

## Communication with NS3

The agent communicates with NS3 via ZeroMQ (ZMQ):

1. **NS3 → Agent**: Sends state information (JSON)
2. **Agent → NS3**: Sends selected action (beaconHz, txPower)
3. **NS3 → Agent**: Sends reward and done flag
4. **Agent → NS3**: Sends acknowledgment

## WandB Integration

To use Weights & Biases for experiment tracking:

```bash
# Install wandb
pip install wandb

# Login (first time only)
wandb login

# Run with wandb logging
python ppo_dual_continuous.py --train --wandb --max_steps 1000
```

Logged metrics:
- Step-by-step rewards
- Total cumulative reward
- PPO loss values
- Action choices
- State features (PDR, CBR, throughput, etc.)
- Number of updates

## Tips for Better Performance

1. **Update Interval**: 
   - Smaller (64-128): Faster learning, less stable
   - Larger (256-512): More stable, slower learning

2. **Entropy Coefficient**:
   - Higher (0.01-0.05): More exploration
   - Lower (0.001-0.01): More exploitation

3. **Clipping Parameter**:
   - Standard value: 0.2
   - More conservative: 0.1
   - More aggressive: 0.3

4. **Learning Rate**:
   - Start with: 0.0003
   - Decrease if unstable: 0.0001
   - Increase if too slow: 0.001

## Troubleshooting

### Agent not learning
- Check if rewards are being received correctly
- Verify state normalization is appropriate
- Try increasing update_interval for more stable updates
- Check entropy coefficient (should encourage exploration)

### Training is unstable
- Reduce learning rate
- Increase update_interval
- Reduce eps_clip parameter
- Add gradient clipping (already included)

### Connection issues
- Ensure NS3 simulation is running
- Check ZMQ port matches (default: 5555)
- Verify firewall settings

## Comparison with DQN Agent

| Feature | PPO | DQN |
|---------|-----|-----|
| Learning Type | On-policy | Off-policy |
| Memory | Trajectory buffer | Experience replay |
| Update Frequency | Every N timesteps | Every step |
| Stability | More stable | Can be unstable |
| Sample Efficiency | Moderate | High (with replay) |
| Exploration | Policy entropy | Epsilon-greedy |

## Example Training Session

```bash
$ python ppo_dual_continuous.py --train --max_steps 1000 --wandb

======================================================================
DUAL-CONTROL CONTINUOUS PPO AGENT FOR VANET
Controls: BeaconHz + TxPower
======================================================================
Mode: TRAINING (Learning continuously)
Port: 5555
Action Space: 36 actions (6 BeaconHz × 6 TxPower)
BeaconHz Options: [2, 4, 6, 8, 10, 12] Hz
TxPower Options: [15, 18, 21, 23, 26, 30] dBm
Max Steps: 1000
Update Interval: 128 timesteps
Device: cuda
======================================================================

[State] Time: 0.5s | PDR: 0.850 | Neighbors: 5.2 | Throughput: 45000 bps | CBR: 0.12
[Current] BeaconHz: 8.0 Hz | TxPower: 23.0 dBm
[Action] BeaconHz: 6 Hz | TxPower: 21 dBm | (Action #8)
[Learning] Memory: 45/128 | Updates: 0
[Feedback] Reward: 0.234

...

[Training] PPO Update #1 | Loss: 0.0456

...

**********************************************************************
PROGRESS SUMMARY (Last 50 steps)
**********************************************************************
Total Steps: 50
Total Reward: 12.45
Avg Recent Reward: 0.249
Avg Recent Loss: 0.0412
PPO Updates: 2
**********************************************************************
```

## License

Part of the VANET RL project.

## Contact

For issues or questions, please refer to the main project documentation.
