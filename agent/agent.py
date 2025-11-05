import zmq
import json

ctx = zmq.Context()
socket = ctx.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("[Agent] Waiting for connection from NS3...")

while True:
    try:
        # === Receive message ===
        msg = socket.recv()
        data = json.loads(msg.decode())

        # print(data)

        # Auto-handle state as dictionary
        state = data.get("data", {})
        time = state.get("time", 0.0)

        print(f"[Agent] t={time:.2f}s State received:")
        # print("State JSON:")
        # print(json.dumps(state, indent=4))
        for k, v in state.items():
            print(f"    {k}: {v}")

        # === Decide action ===
        # Example: simple rule (replace with your policy or NN)
        if state.get("PDR", 0) < 0.8:
            action = {"beaconHz": 12}
        else:
            action = {"beaconHz": 8}

        socket.send_string(json.dumps({"action": action}))

        # === Receive reward ===
        msg = socket.recv()
        reward_msg = json.loads(msg.decode())
        reward = reward_msg.get("reward", 0.0)
        print(f"[Agent] Reward: {reward}\n")

        # Send ACK
        socket.send_string("ack")

    except KeyboardInterrupt:
        print("\n[Agent] Stopped.")
        break
