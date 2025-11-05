run the python agent
then run the ns3 script using
./ns3 build
./build/scratch/v2x/scratch_v2x_v2x --lanes=6 --vehPerLane=300 --simTime=10 --beaconHz=10 --payloadBytes=1200 --txPowerDbm=23 --rtsCts=1 --flowmon=1
used ns3 version 3.40