import zmq
import json

ctx = zmq.Context()
socket = ctx.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("[Agent] Waiting for connection from NS3...")

while True:
    try:
        # === 1. Receive state ===
        msg = socket.recv()
        data = msg.decode()

        # Handle both plain or JSON-based state
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "state" in parsed:
                state = parsed["state"]
            else:
                state = [float(parsed)]
        except json.JSONDecodeError:
            state = [float(data)]

        metric = state[0]
        print(f"[Agent] State received: metric={metric}")

        # === 2. Compute action ===
        if metric < 1000:
            action = 2
        elif metric > 3000:
            action = 0
        else:
            action = 1

        # === 3. Send action (now as JSON) ===
        socket.send_string(json.dumps({"action": action}))
        print(f"[Agent] Sent action: {action}")

        # === 4. Receive reward ===
        msg = socket.recv()
        reward_data = msg.decode()

        try:
            reward_parsed = json.loads(reward_data)
            reward = reward_parsed.get("reward", reward_data)
        except json.JSONDecodeError:
            reward = float(reward_data.split(",")[0])

        print(f"[Agent] Reward received: {reward}")

        # === 5. Send acknowledgment to keep ZMQ REQ/REP cycle in sync ===
        socket.send_string("ack")

    except KeyboardInterrupt:
        print("\n[Agent] Interrupted by user.")
        break
    except Exception as e:
        print(f"[Agent] Error: {e}")
        try:
            socket.send_string("error")
        except:
            pass
