run the python agent
then run the ns3 script using
./ns3 build
./build/scratch/v2x/scratch_v2x_v2x --lanes=6 --vehPerLane=300 --simTime=10 --beaconHz=10 --payloadBytes=1200 --txPowerDbm=23 --rtsCts=1 --flowmon=1
used ns3 version 3.40

//rl setup

run python script

python dqn_simple_continuous.py --train --max_steps 0 --save_interval 100

python -u dqn_simple_continuous.py --train --max_steps 0 --save_interval 100 > training.log 2>&1

python dqn_simple_continuous.py --train --max_steps 500 --wandb

run the vanet

./build/scratch/v2x/scratch_v2x_v2x --vehicles=100 --simTime=7200 --logInterval=5.0 --beaconHz=2.0 --enableRL=1

wandb api key - 

running the vanet sumo
./build/scratch/v2x_sumo/scratch_v2x_sumo_vanet \
  --simTime=36000 \
  --logInterval=5.0 \
  --beaconHz=2.0 \
  --enableRL=1

running the updated vanet with the new pdr calculation

