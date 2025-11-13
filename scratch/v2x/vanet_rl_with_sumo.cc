/*
 * VANET Scenario for Reinforcement Learning with SUMO Mobility
 * Compatible with NS-3 3.40
 * 
 * Uses SUMO-generated mobility trace in NS-2 TCL format
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

NS_LOG_COMPONENT_DEFINE("VanetRLSumo");

// Global variables
double g_beaconInterval = 1.0;
double g_txPower = 23.0;
uint32_t g_numVehicles = 100;  // Will be overridden by TCL file
double g_simulationTime = 200.0;
double g_loggingInterval = 5.0;
bool g_enableRL = false;
std::string g_rlAddress = "tcp://localhost:5555";
std::string g_traceFile = "/home/ubuntu/sumo/ns3mobility.tcl";

std::map<uint32_t, uint64_t> g_packetsSent;
std::map<uint32_t, uint64_t> g_packetsReceived;
// NEW: Track realistic expected receptions based on neighbors in range
std::map<uint32_t, uint64_t> g_expectedReceptions;  // How many packets this node SHOULD receive
// NEW: Track channel busy time for CBR calculation
std::map<uint32_t, Time> g_channelBusyTime;  // Total time channel is busy per node
std::map<uint32_t, Time> g_lastChannelSampleTime;  // Last time CBR was sampled
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
    
    double commRange = 300.0; // Match RangePropagationLossModel MaxRange
    
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
        if (distance <= 300.0) {  // FIX: Match communication range (was 500.0)
            totalDistance += distance;
            count++;
        }
    }
    return count > 0 ? totalDistance / count : 0.0;
}

// NEW: Calculate Channel Busy Ratio (CBR) for a node
double CalculateCBR(Ptr<Node> node) {
    uint32_t nodeId = node->GetId();
    Ptr<NetDevice> device = node->GetDevice(0);
    Ptr<WifiNetDevice> wifiDevice = DynamicCast<WifiNetDevice>(device);
    
    if (!wifiDevice) {
        return 0.0;
    }
    
    Ptr<WifiPhy> wifiPhy = wifiDevice->GetPhy();
    if (!wifiPhy) {
        return 0.0;
    }
    
    // Get current time
    Time currentTime = Simulator::Now();
    
    // Get time since last sample
    Time timeSinceLastSample = currentTime - g_lastChannelSampleTime[nodeId];
    
    if (timeSinceLastSample.GetSeconds() == 0.0) {
        return 0.0;
    }
    
    // In NS-3, we can estimate CBR by tracking PHY state
    // CBR = (time in TX + time in RX + time in CCA_BUSY) / total_time
    // For simplicity, we'll estimate based on packet activity
    
    // Alternative: Use channel activity metrics
    // Here we use a simplified approach based on transmission activity
    Time busyTime = wifiPhy->GetDelayUntilIdle();
    
    // Update last sample time
    g_lastChannelSampleTime[nodeId] = currentTime;
    
    // Return CBR (0.0 to 1.0)
    return std::min(1.0, busyTime.GetSeconds());
}

// NEW: Get average CBR across all nodes
double GetAverageCBR() {
    double totalCBR = 0.0;
    uint32_t nodeCount = 0;
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        Ptr<Node> node = g_nodes.Get(i);
        double cbr = CalculateCBR(node);
        totalCBR += cbr;
        nodeCount++;
    }
    
    return (nodeCount > 0) ? totalCBR / nodeCount : 0.0;
}

// Forward declarations
void UpdateBeaconInterval(double newInterval);
void UpdateTxPower(double newPower);

// Calculate reward based on network performance
double CalculateReward(double pdr, double throughput, double avgNeighbors) {
    // IMPROVED REWARD FUNCTION
    // Goal: High PDR, Good throughput, Reasonable beaconHz
    
    double beaconHz = 1.0 / g_beaconInterval;
    
    // 1. PDR reward (most important): 0-40 points
    double pdrReward = pdr * 40.0;
    
    // 2. Throughput reward: Optimal around 20-30 kbps
    // Too low = bad (poor connectivity)
    // Too high = bad (network congestion)
    double targetThroughput = 25000.0; // 25 kbps target
    double throughputRatio = throughput / targetThroughput;
    double throughputReward = 0.0;
    
    if (throughputRatio < 0.5) {
        // Too low throughput
        throughputReward = throughputRatio * 20.0; // 0-10 points
    } else if (throughputRatio <= 1.5) {
        // Good throughput range
        throughputReward = 10.0 + (1.0 - std::abs(throughputRatio - 1.0)) * 10.0; // 10-20 points
    } else {
        // Too high (congestion)
        throughputReward = std::max(0.0, 20.0 - (throughputRatio - 1.5) * 10.0);
    }
    
    // 3. BeaconHz reward: Optimal around 6-10 Hz
    // Too low (2-4 Hz) = missed safety messages
    // Too high (12+ Hz) = waste energy, cause interference
    double beaconReward = 0.0;
    
    if (beaconHz < 4.0) {
        // Too low - dangerous
        beaconReward = beaconHz * 2.5; // 0-10 points
    } else if (beaconHz <= 10.0) {
        // Optimal range
        beaconReward = 10.0 + (beaconHz - 4.0) * 1.67; // 10-20 points
    } else {
        // Too high - wasteful
        beaconReward = std::max(0.0, 20.0 - (beaconHz - 10.0) * 2.0);
    }
    
    // 4. Connectivity reward: More neighbors = better
    double connectivityReward = std::min(avgNeighbors / 10.0 * 10.0, 10.0); // 0-10 points
    
    // 5. Stability bonus: Penalize extreme values
    double stabilityPenalty = 0.0;
    if (pdr < 0.5) stabilityPenalty += 5.0;  // Very bad PDR
    if (avgNeighbors < 3.0) stabilityPenalty += 5.0;  // Too isolated
    
    // Total reward: 0-100 points possible
    double totalReward = pdrReward + throughputReward + beaconReward + 
                        connectivityReward - stabilityPenalty;
    
    // Ensure reward is always positive
    totalReward = std::max(0.0, totalReward);
    
    // Debug output
    std::cout << "[Reward Breakdown]" << std::endl;
    std::cout << "  PDR:          " << std::fixed << std::setprecision(2) << pdrReward << std::endl;
    std::cout << "  Throughput:   " << throughputReward << std::endl;
    std::cout << "  BeaconHz:     " << beaconReward << " (current: " << beaconHz << " Hz)" << std::endl;
    std::cout << "  Connectivity: " << connectivityReward << std::endl;
    std::cout << "  Penalty:      -" << stabilityPenalty << std::endl;
    std::cout << "  TOTAL:        " << totalReward << std::endl;
    
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
    
    // Only log first 10 nodes to reduce output with large SUMO traces
    uint32_t nodesToLog = std::min((uint32_t)10, g_nodes.GetN());
    
    for (uint32_t i = 0; i < nodesToLog; i++) {
        Ptr<Node> node = g_nodes.Get(i);
        uint32_t nodeId = node->GetId();
        
        uint64_t sent = g_packetsSent[nodeId];
        uint64_t recv = g_packetsReceived[nodeId];
        uint64_t expectedRecv = g_expectedReceptions[nodeId]; // NEW: realistic expectation
        
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
                  << " | Pos(" << std::setw(6) << (int)pos.x << "," << std::setw(6) << (int)pos.y << ")"
                  << " Vel(" << std::setw(4) << (int)vel.x << "," << std::setw(4) << (int)vel.y << ")"
                  << " Neigh:" << std::setw(2) << neighbors
                  << " Dist:" << std::setw(4) << (int)avgDist << "m"
                  << std::endl;
    }
    
    if (g_nodes.GetN() > nodesToLog) {
        std::cout << "... (" << (g_nodes.GetN() - nodesToLog) << " more nodes)" << std::endl;
    }
    
    // Calculate total metrics
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        totalSent += g_packetsSent[i];
        totalRecv += g_packetsReceived[i];
        totalExpectedRecv += g_expectedReceptions[i];  // NEW
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
    
    // NEW: Calculate average Channel Busy Ratio
    double avgCBR = GetAverageCBR();
    
    std::cout << "========================================" << std::endl;
    std::cout << "TOTAL Sent:" << totalSent << " Recv:" << totalRecv 
              << " Expected:" << totalExpectedRecv  // NEW: show total expected
              << " GlobalPDR:" << std::fixed << std::setprecision(3) << globalPDR 
              << " AvgNeighbors:" << std::setprecision(1) << avgNeighbors 
              << " CBR:" << std::setprecision(3) << avgCBR << std::endl;  // NEW: show CBR
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
        state["CBR"] = avgCBR;  // NEW: Add Channel Busy Ratio
        
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
    double oldInterval = g_beaconInterval;
    double oldBeaconHz = 1.0 / oldInterval;
    
    g_beaconInterval = newInterval;
    double newBeaconHz = 1.0 / g_beaconInterval;
    
    std::cout << "\n╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  BEACON INTERVAL UPDATE                      ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Time:         " << std::setw(8) << Simulator::Now().GetSeconds() << " s          ║" << std::endl;
    std::cout << "║  Old Interval: " << std::setw(8) << oldInterval << " s (" << std::setw(4) << oldBeaconHz << " Hz) ║" << std::endl;
    std::cout << "║  New Interval: " << std::setw(8) << newInterval << " s (" << std::setw(4) << newBeaconHz << " Hz) ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝" << std::endl;
    
    // Update all OnOff applications with new data rate
    for (uint32_t i = 0; i < g_onoffApps.GetN(); i++) {
        Ptr<Application> app = g_onoffApps.Get(i);
        Ptr<OnOffApplication> onoff = DynamicCast<OnOffApplication>(app);
        
        if (onoff) {
            // Calculate new data rate: packet_size(bytes) * 8 / interval(s) = bits/s
            DataRate newRate((200 * 8) / g_beaconInterval);
            onoff->SetAttribute("DataRate", DataRateValue(newRate));
        }
    }
    std::cout << "  ✓ Updated " << g_onoffApps.GetN() << " node applications" << std::endl;
    std::cout << std::endl;
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
    cmd.AddValue("vehicles", "Number of vehicles (overridden by trace file)", g_numVehicles);
    cmd.AddValue("simTime", "Simulation time", g_simulationTime);
    cmd.AddValue("beaconHz", "Beacon frequency Hz", g_beaconInterval);
    cmd.AddValue("txPower", "TX power dBm", g_txPower);
    cmd.AddValue("logInterval", "Log interval", g_loggingInterval);
    cmd.AddValue("enableRL", "Enable RL agent communication", g_enableRL);
    cmd.AddValue("rlAddress", "RL agent ZMQ address", g_rlAddress);
    cmd.AddValue("traceFile", "SUMO mobility trace file (TCL format)", g_traceFile);
    cmd.Parse(argc, argv);
    
    double beaconHz = g_beaconInterval;
    g_beaconInterval = 1.0 / beaconHz;
    
    std::cout << "\n=========================================\n";
    std::cout << "VANET RL with SUMO Mobility\n";
    std::cout << "=========================================\n";
    std::cout << "Trace File: " << g_traceFile << "\n";
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
    
    // Create nodes - will be populated by Ns2MobilityHelper
    // First, count nodes in trace file
    std::ifstream traceStream(g_traceFile);
    if (!traceStream.is_open()) {
        std::cerr << "ERROR: Cannot open trace file: " << g_traceFile << std::endl;
        return 1;
    }
    
    uint32_t maxNodeId = 0;
    std::string line;
    while (std::getline(traceStream, line)) {
        if (line.find("$node_(") != std::string::npos) {
            size_t start = line.find("$node_(") + 7;
            size_t end = line.find(")", start);
            if (end != std::string::npos) {
                uint32_t nodeId = std::stoul(line.substr(start, end - start));
                if (nodeId > maxNodeId) {
                    maxNodeId = nodeId;
                }
            }
        }
    }
    traceStream.close();
    
    g_numVehicles = maxNodeId + 1;
    std::cout << "Detected " << g_numVehicles << " vehicles in trace file\n\n";
    
    g_nodes.Create(g_numVehicles);
    
    // Setup mobility using Ns2MobilityHelper
    Ns2MobilityHelper ns2mobility = Ns2MobilityHelper(g_traceFile);
    ns2mobility.Install();
    
    std::cout << "✓ SUMO mobility trace loaded successfully\n\n";
    
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
    
    // Internet stack
    InternetStackHelper internet;
    internet.Install(g_nodes);
    
    // Use /16 subnet to support many vehicles
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
        
        // Sender - OnOff application for periodic beacons
        OnOffHelper onoff("ns3::UdpSocketFactory",
                         InetSocketAddress(Ipv4Address("10.1.255.255"), port));
        onoff.SetConstantRate(DataRate((200 * 8) / g_beaconInterval));
        onoff.SetAttribute("PacketSize", UintegerValue(200));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1000]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        
        ApplicationContainer app = onoff.Install(g_nodes.Get(i));
        app.Start(Seconds(1.0 + i * 0.001)); // Stagger start times slightly
        app.Stop(Seconds(g_simulationTime));
        
        // Store the OnOff app for runtime updates
        g_onoffApps.Add(app);
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
        g_channelBusyTime[i] = Seconds(0.0);  // NEW: initialize CBR tracking
        g_lastChannelSampleTime[i] = Seconds(0.0);  // NEW: initialize CBR tracking
    }
    
    // Schedule logging
    Simulator::Schedule(Seconds(g_loggingInterval), &LogMetrics);
    
    // Dynamic parameter changes (only if RL is disabled)
    if (!g_enableRL) {
        Simulator::Schedule(Seconds(20.0), &UpdateBeaconInterval, 0.2);  // 5 Hz
        Simulator::Schedule(Seconds(40.0), &UpdateTxPower, 26.0);
        Simulator::Schedule(Seconds(60.0), &UpdateBeaconInterval, 0.1); // 10 Hz
    }
    
    std::cout << "Starting simulation with " << g_nodes.GetN() << " vehicles...\n\n";
    
    Simulator::Stop(Seconds(g_simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    std::cout << "\n=== Simulation Complete! ===\n";
    
    return 0;
}
