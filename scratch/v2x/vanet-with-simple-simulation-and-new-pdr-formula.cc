/*
 * VANET Scenario for Reinforcement Learning
 * Compatible with NS-3 3.40
 * 
 * Integrated with RL agent via ZMQ
 * Sends state, receives actions (beaconHz, txPower), sends rewards
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "rl-interface.h"
#include <iostream>
#include <iomanip>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VanetRL");

// Global variables
double g_beaconInterval = 1.0;
double g_txPower = 23.0;
uint32_t g_numVehicles = 20;
double g_simulationTime = 100.0;
double g_loggingInterval = 5.0;
bool g_enableRL = false;
std::string g_rlAddress = "tcp://localhost:5555";

std::map<uint32_t, uint64_t> g_packetsSent;
std::map<uint32_t, uint64_t> g_packetsReceived;
// NEW: Track realistic expected receptions based on neighbors in range
std::map<uint32_t, uint64_t> g_expectedReceptions;  // How many packets this node SHOULD receive
NodeContainer g_nodes;
NetDeviceContainer g_devices;
ApplicationContainer g_onoffApps;  // Store OnOff apps for runtime updates
Ptr<RLInterface> g_rlInterface;

// Metrics for reward calculation
double g_previousPDR = 0.0;
double g_previousThroughput = 0.0;

// NEW: Track neighbors in range at transmission time
void UpdateExpectedReceptions(uint32_t senderNodeId) {
    // When a node sends a packet, count how many nodes are in range
    // Those nodes SHOULD receive it (realistic denominator for PDR)
    Ptr<Node> senderNode = g_nodes.Get(senderNodeId);
    Ptr<MobilityModel> senderMobility = senderNode->GetObject<MobilityModel>();
    
    double commRange = 300.0; // Match your RangePropagationLossModel MaxRange
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        if (i == senderNodeId) continue; // Don't count sender
        
        Ptr<MobilityModel> receiverMobility = g_nodes.Get(i)->GetObject<MobilityModel>();
        double distance = senderMobility->GetDistanceFrom(receiverMobility);
        
        if (distance <= commRange) {
            // This node is in range, so it SHOULD receive this packet
            g_expectedReceptions[i]++;
        }
    }
}

// Packet sent callback
void TxCallback(std::string context, Ptr<const Packet> packet) {
    // Context format: /NodeList/<nodeId>/ApplicationList/...
    size_t pos = context.find("/NodeList/");
    if (pos != std::string::npos) {
        size_t start = pos + 10; // Length of "/NodeList/"
        size_t end = context.find("/", start);
        if (end != std::string::npos) {
            uint32_t nodeId = std::stoul(context.substr(start, end - start));
            g_packetsSent[nodeId]++;
            // NEW: Update expected receptions for all nodes in range
            UpdateExpectedReceptions(nodeId);
        }
    }
}

// Packet received callback
void RxCallback(std::string context, Ptr<const Packet> packet, const Address &address) {
    // Context format: /NodeList/<nodeId>/ApplicationList/...
    size_t pos = context.find("/NodeList/");
    if (pos != std::string::npos) {
        size_t start = pos + 10; // Length of "/NodeList/"
        size_t end = context.find("/", start);
        if (end != std::string::npos) {
            uint32_t nodeId = std::stoul(context.substr(start, end - start));
            g_packetsReceived[nodeId]++;
        }
    }
}

// Calculate neighbors
uint32_t CountNeighbors(Ptr<Node> node, double range) {
    uint32_t count = 0;
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        if (g_nodes.Get(i)->GetId() == node->GetId()) continue;
        Ptr<MobilityModel> otherMobility = g_nodes.Get(i)->GetObject<MobilityModel>();
        if (mobility->GetDistanceFrom(otherMobility) <= range) count++;
    }
    return count;
}

// Average inter-vehicle distance
double GetAvgInterVehicleDistance(Ptr<Node> node) {
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    double totalDistance = 0.0;
    uint32_t count = 0;
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        if (g_nodes.Get(i)->GetId() == node->GetId()) continue;
        Ptr<MobilityModel> otherMobility = g_nodes.Get(i)->GetObject<MobilityModel>();
        double distance = mobility->GetDistanceFrom(otherMobility);
        if (distance <= 500.0) {
            totalDistance += distance;
            count++;
        }
    }
    return count > 0 ? totalDistance / count : 0.0;
}

// Forward declarations
void UpdateBeaconInterval(double newInterval);
void UpdateTxPower(double newPower);

// Calculate reward based on network performance
double CalculateReward(double pdr, double throughput, double avgNeighbors) {
    // Reward components:
    // 1. PDR should be high (0-1)
    // 2. Throughput should be reasonable but not excessive
    // 3. Energy efficiency (inversely related to beaconHz and txPower)
    
    double pdrReward = pdr * 10.0;  // Max 10 points for perfect PDR
    
    // Throughput reward: target around 50-100 kbps, penalize too high or too low
    double targetThroughput = 75000.0; // 75 kbps
    double throughputError = std::abs(throughput - targetThroughput) / targetThroughput;
    double throughputReward = 5.0 * (1.0 - throughputError);
    if (throughputReward < 0) throughputReward = 0;
    
    // Energy efficiency: penalize high beacon rate and high power
    double beaconHz = 1.0 / g_beaconInterval;
    double energyCost = (beaconHz / 20.0) + ((g_txPower - 20.0) / 10.0);
    double energyReward = 5.0 - energyCost;
    if (energyReward < 0) energyReward = 0;
    
    // Neighbor connectivity reward
    double neighborReward = std::min(avgNeighbors / 10.0, 3.0);
    
    double totalReward = pdrReward + throughputReward + energyReward + neighborReward;
    
    return totalReward;
}

// Log metrics and interact with RL agent
void LogMetrics() {
    double currentTime = Simulator::Now().GetSeconds();
    
    std::cout << "========================================" << std::endl;
    std::cout << "Time: " << currentTime << "s | BeaconHz: " << (1.0/g_beaconInterval)
              << " | TxPower: " << g_txPower << " dBm" << std::endl;
    std::cout << "========================================" << std::endl;
    
    uint64_t totalSent = 0, totalRecv = 0;
    uint64_t totalExpectedRecv = 0; // NEW: realistic expected receptions
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        Ptr<Node> node = g_nodes.Get(i);
        uint32_t nodeId = node->GetId();
        
        uint64_t sent = g_packetsSent[nodeId];
        uint64_t recv = g_packetsReceived[nodeId];
        uint64_t expectedRecv = g_expectedReceptions[nodeId]; // NEW: realistic expectation
        totalSent += sent;
        totalRecv += recv;
        totalExpectedRecv += expectedRecv;
        
        // NEW: PDR based on realistic expected receptions (only from nodes in range)
        double nodePDR = (expectedRecv > 0) ? (double)recv / expectedRecv : 0.0;
        
        double throughput = (recv * 200 * 8) / currentTime;
        
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
        Vector pos = mobility->GetPosition();
        Vector vel = mobility->GetVelocity();
        
        uint32_t neighbors = CountNeighbors(node, 300.0);
        double avgDist = GetAvgInterVehicleDistance(node);
        
        std::cout << "Node[" << std::setw(2) << nodeId << "]"
                  << " Sent:" << std::setw(5) << sent
                  << " Recv:" << std::setw(5) << recv
                  << " Exp:" << std::setw(5) << expectedRecv  // NEW: show expected
                  << " PDR:" << std::fixed << std::setprecision(3) << std::setw(6) << nodePDR
                  << " Tput:" << std::setw(8) << (int)throughput << "bps"
                  << " | Pos(" << std::setw(4) << (int)pos.x << "," << std::setw(2) << (int)pos.y << ")"
                  << " Vel(" << std::setw(2) << (int)vel.x << ",0)"
                  << " Neigh:" << std::setw(2) << neighbors
                  << " Dist:" << std::setw(4) << (int)avgDist << "m"
                  << std::endl;
    }
    
    // NEW: Global PDR using realistic expected receptions
    double globalPDR = (totalExpectedRecv > 0) ? (double)totalRecv / totalExpectedRecv : 0.0;
    
    // Calculate average metrics
    double avgThroughput = 0.0, avgNeighbors = 0.0;
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        Ptr<Node> node = g_nodes.Get(i);
        uint32_t nodeId = node->GetId();
        uint64_t recv = g_packetsReceived[nodeId];
        double throughput = (recv * 200 * 8) / currentTime;
        avgThroughput += throughput;
        avgNeighbors += CountNeighbors(node, 300.0);
    }
    avgThroughput /= g_nodes.GetN();
    avgNeighbors /= g_nodes.GetN();
    
    std::cout << "========================================" << std::endl;
    std::cout << "TOTAL Sent:" << totalSent << " Recv:" << totalRecv 
              << " Expected:" << totalExpectedRecv  // NEW: show total expected
              << " GlobalPDR:" << std::fixed << std::setprecision(3) << globalPDR 
              << " AvgNeighbors:" << std::setprecision(1) << avgNeighbors << std::endl;
    std::cout << std::endl;
    
    // RL Agent Interaction
    if (g_enableRL && g_rlInterface) {
        // Build state dictionary
        std::map<std::string, double> state;
        state["time"] = currentTime;
        state["PDR"] = globalPDR;
        state["throughput"] = avgThroughput;
        state["packetsSent"] = totalSent;
        state["packetsReceived"] = totalRecv;
        state["avgNeighbors"] = avgNeighbors;
        state["beaconHz"] = 1.0 / g_beaconInterval;
        state["txPower"] = g_txPower;
        state["numVehicles"] = g_nodes.GetN();
        
        // Send state to RL agent
        g_rlInterface->SendState(state);
        
        // Receive action from RL agent
        nlohmann::json actionJson = g_rlInterface->ReceiveAction();
        
        // Apply action
        if (actionJson.contains("action")) {
            auto action = actionJson["action"];
            
            if (action.contains("beaconHz")) {
                double newBeaconHz = action["beaconHz"];
                if (newBeaconHz > 0 && newBeaconHz <= 20.0) {
                    UpdateBeaconInterval(1.0 / newBeaconHz);
                }
            }
            
            if (action.contains("txPower")) {
                double newTxPower = action["txPower"];
                if (newTxPower >= 10.0 && newTxPower <= 30.0) {
                    UpdateTxPower(newTxPower);
                }
            }
        }
        
        // Calculate reward
        double reward = CalculateReward(globalPDR, avgThroughput, avgNeighbors);
        
        // Check if simulation is done
        bool done = (currentTime >= g_simulationTime - g_loggingInterval - 0.1);
        
        // Send reward to RL agent
        g_rlInterface->SendReward(reward, done);
        
        std::cout << ">>> RL Reward: " << std::fixed << std::setprecision(3) 
                  << reward << " | Done: " << (done ? "Yes" : "No") << " <<<" << std::endl;
        std::cout << std::endl;
        
        g_previousPDR = globalPDR;
        g_previousThroughput = avgThroughput;
    }
    
    if (currentTime < g_simulationTime - 0.1) {
        Simulator::Schedule(Seconds(g_loggingInterval), &LogMetrics);
    }
}

// Update beacon interval
void UpdateBeaconInterval(double newInterval) {
    g_beaconInterval = newInterval;
    std::cout << ">>> Beacon interval updated to " << g_beaconInterval 
              << "s (" << (1.0/g_beaconInterval) << " Hz) <<<" << std::endl;
}

// Update TX power
void UpdateTxPower(double newPower) {
    g_txPower = newPower;
    for (uint32_t i = 0; i < g_devices.GetN(); i++) {
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(g_devices.Get(i));
        if (wifiDev) {
            wifiDev->GetPhy()->SetTxPowerStart(newPower);
            wifiDev->GetPhy()->SetTxPowerEnd(newPower);
        }
    }
    std::cout << ">>> TX Power updated to " << g_txPower << " dBm <<<" << std::endl;
}

int main(int argc, char *argv[]) {
    CommandLine cmd;
    cmd.AddValue("vehicles", "Number of vehicles", g_numVehicles);
    cmd.AddValue("simTime", "Simulation time", g_simulationTime);
    cmd.AddValue("beaconHz", "Beacon frequency Hz", g_beaconInterval);
    cmd.AddValue("txPower", "TX power dBm", g_txPower);
    cmd.AddValue("logInterval", "Log interval", g_loggingInterval);
    cmd.AddValue("enableRL", "Enable RL agent communication", g_enableRL);
    cmd.AddValue("rlAddress", "RL agent ZMQ address", g_rlAddress);
    cmd.Parse(argc, argv);
    
    double beaconHz = g_beaconInterval;
    g_beaconInterval = 1.0 / beaconHz;
    
    std::cout << "\n=========================================\n";
    std::cout << "VANET RL Environment Simulation\n";
    std::cout << "=========================================\n";
    std::cout << "Vehicles: " << g_numVehicles << "\n";
    std::cout << "SimTime: " << g_simulationTime << "s\n";
    std::cout << "BeaconHz: " << beaconHz << "\n";
    std::cout << "TxPower: " << g_txPower << " dBm\n";
    std::cout << "RL Enabled: " << (g_enableRL ? "Yes" : "No") << "\n";
    if (g_enableRL) {
        std::cout << "RL Address: " << g_rlAddress << "\n";
    }
    std::cout << "=========================================\n\n";
    
    // Initialize RL interface if enabled
    if (g_enableRL) {
        g_rlInterface = CreateObject<RLInterface>();
        g_rlInterface->Init(g_rlAddress);
    }
    
    // Create nodes
    g_nodes.Create(g_numVehicles);
    
    // WiFi setup
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel",
                                    "MaxRange", DoubleValue(300.0));
    
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(g_txPower));
    wifiPhy.Set("TxPowerEnd", DoubleValue(g_txPower));
    
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                  "DataMode", StringValue("OfdmRate6MbpsBW10MHz"));
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    
    g_devices = wifi.Install(wifiPhy, wifiMac, g_nodes);
    
    // Mobility
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
    
    for (uint32_t i = 0; i < g_numVehicles; i++) {
        uint32_t lane = i % 4;
        double x = (i / 4) * 50.0;
        double y = lane * 3.5;
        posAlloc->Add(Vector(x, y, 0.0));
    }
    
    mobility.SetPositionAllocator(posAlloc);
    mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    mobility.Install(g_nodes);
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        Ptr<ConstantVelocityMobilityModel> mob = 
            g_nodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
        double speed = 25.0 + (rand() % 3);
        mob->SetVelocity(Vector(speed, 0.0, 0.0));
    }
    
    // Internet stack
    InternetStackHelper internet;
    internet.Install(g_nodes);
    
    // Use /16 subnet to support up to ~65,000 vehicles
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.0.0", "255.255.0.0");
    ipv4.Assign(g_devices);
    
    // Applications - Simple OnOff to broadcast address
    uint16_t port = 9;
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        // Receiver
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sinkHelper.Install(g_nodes.Get(i));
        sinkApp.Start(Seconds(0.0));
        sinkApp.Stop(Seconds(g_simulationTime));
        
        // Sender
        OnOffHelper onoff("ns3::UdpSocketFactory",
                         InetSocketAddress(Ipv4Address("10.1.255.255"), port));
        onoff.SetConstantRate(DataRate((200 * 8) / g_beaconInterval));
        onoff.SetAttribute("PacketSize", UintegerValue(200));
        
        ApplicationContainer app = onoff.Install(g_nodes.Get(i));
        app.Start(Seconds(1.0 + i * 0.01));
        app.Stop(Seconds(g_simulationTime));
    }
    
    // Connect traces
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::OnOffApplication/Tx",
                    MakeCallback(&TxCallback));
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx",
                    MakeCallback(&RxCallback));
    
    // Initialize metrics
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        g_packetsSent[i] = 0;
        g_packetsReceived[i] = 0;
        g_expectedReceptions[i] = 0;  // NEW: initialize expected receptions
    }
    
    // Schedule logging
    Simulator::Schedule(Seconds(g_loggingInterval), &LogMetrics);
    
    // Dynamic parameter changes (only if RL is disabled)
    if (!g_enableRL) {
        Simulator::Schedule(Seconds(20.0), &UpdateBeaconInterval, 0.2);  // 5 Hz
        Simulator::Schedule(Seconds(40.0), &UpdateTxPower, 26.0);
        Simulator::Schedule(Seconds(60.0), &UpdateBeaconInterval, 0.1); // 10 Hz
    }
    
    std::cout << "Starting simulation...\n\n";
    
    Simulator::Stop(Seconds(g_simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    std::cout << "\n=== Simulation Complete! ===\n";
    
    return 0;
}
