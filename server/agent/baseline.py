"""
Simple Baseline Logger for NS-3 VANET
No RL - just logs metrics to WandB without changing parameters
"""

import zmq
import json
import wandb
import argparse
from datetime import datetime


def run_baseline_logger(port=5555, project_name="vanet-baseline", run_name=None):
    """
    Simple logger that:
    1. Receives state from NS-3
    2. Sends back the SAME beaconHz and txPower (no changes)
    3. Logs all metrics to WandB
    """
    
    # Initialize WandB
    if run_name is None:
        run_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "mode": "baseline",
            "no_rl": True,
            "description": "Baseline performance without RL control"
        }
    )
    
    # Setup ZMQ to communicate with NS-3
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    print("\n" + "="*60)
    print("VANET Baseline Logger (No RL)")
    print("="*60)
    print(f"Project: {project_name}")
    print(f"Run: {run_name}")
    print(f"Port: {port}")
    print("="*60)
    print("\nWaiting for NS-3 simulation...\n")
    
    step = 0
    total_reward = 0.0
    
    try:
        while True:
            # Receive message from NS-3
            msg = socket.recv()
            data = json.loads(msg.decode())
            
            # Check message type
            msg_type = data.get('type', 'unknown')
            
            if msg_type == 'state':
                # Extract state
                state = data.get('data', {})
                
                # Get current parameters (to send back unchanged)
                current_beaconHz = state.get('beaconHz', 8.0)
                current_txPower = state.get('txPower', 23.0)
                
                # Print state
                print(f"[Step {step}] Time: {state.get('time', 0):.1f}s | "
                      f"PDR: {state.get('PDR', 0):.3f} | "
                      f"Throughput: {state.get('throughput', 0):.0f} bps | "
                      f"Neighbors: {state.get('avgNeighbors', 0):.1f} | "
                      f"CBR: {state.get('CBR', 0):.3f}")
                
                # Send back SAME parameters (no changes!)
                response = {
                    "action": {
                        "beaconHz": current_beaconHz,
                        "txPower": current_txPower
                    }
                }
                socket.send_string(json.dumps(response))
                
                # Wait for reward
                msg = socket.recv()
                reward_data = json.loads(msg.decode())
                reward = reward_data.get('reward', 0.0)
                done = reward_data.get('done', False)
                
                total_reward += reward
                
                print(f"         Reward: {reward:.2f} | Total: {total_reward:.2f}")
                
                # Log everything to WandB
                wandb.log({
                    # Timestep
                    "step": step,
                    "time": state.get('time', 0),
                    
                    # Network metrics (same names as PPO agent)
                    "state_pdr": state.get('PDR', 0),
                    "state_throughput": state.get('throughput', 0),
                    "state_avg_neighbors": state.get('avgNeighbors', 0),
                    "state_cbr": state.get('CBR', 0),
                    
                    # Packet stats
                    "packets_sent": state.get('packetsSent', 0),
                    "packets_received": state.get('packetsReceived', 0),
                    
                    # Parameters (constant in baseline)
                    "state_beaconHz": current_beaconHz,
                    "state_txPower": current_txPower,
                    "num_vehicles": state.get('numVehicles', 0),
                    
                    # Reward
                    "reward": reward,
                    "total_reward": total_reward,
                })
                
                # Send acknowledgment
                socket.send_string("ack")
                
                step += 1
                
                # Check if done
                if done:
                    print("\n[INFO] Simulation complete (done=True)")
                    break
            
            else:
                # Unknown message type, send default response
                socket.send_string(json.dumps({
                    "action": {"beaconHz": 8.0, "txPower": 23.0}
                }))
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user (Ctrl+C)")
    
    finally:
        print("\n" + "="*60)
        print("BASELINE RUN SUMMARY")
        print("="*60)
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Average Reward: {total_reward/step if step > 0 else 0:.2f}")
        print("="*60)
        
        wandb.finish()
        socket.close()
        ctx.term()
        
        print(f"\n✅ Results logged to WandB project: {project_name}")
        print(f"   Run name: {run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline Logger for VANET (No RL)')
    parser.add_argument('--port', type=int, default=5555,
                       help='ZMQ port (default: 5555)')
    parser.add_argument('--project', type=str, default='vanet-baseline',
                       help='WandB project name (default: vanet-baseline)')
    parser.add_argument('--name', type=str, default=None,
                       help='WandB run name (default: auto-generated)')
    
    args = parser.parse_args()
    
    run_baseline_logger(
        port=args.port,
        project_name=args.project,
        run_name=args.name
    )