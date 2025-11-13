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

// NEW: Windowed metrics for continuous learning
struct MetricsWindow {
    uint64_t packetsSent = 0;
    uint64_t packetsReceived = 0;
    uint64_t expectedReceptions = 0;
    Time windowStart = Seconds(0);
};

std::map<uint32_t, MetricsWindow> g_currentWindow;  // Current measurement window
std::map<uint32_t, MetricsWindow> g_previousWindow; // Previous window (for comparison)

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
            g_currentWindow[i].expectedReceptions++;  // Track in current window
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
            g_currentWindow[nodeId].packetsSent++;  // Track in current window
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
            g_currentWindow[nodeId].packetsReceived++;  // Track in current window
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
    
    Time currentTime = Simulator::Now();
    Time timeSinceLastSample = currentTime - g_lastChannelSampleTime[nodeId];
    
    if (timeSinceLastSample.GetSeconds() <= 0.0) {
        g_lastChannelSampleTime[nodeId] = currentTime;
        return 0.0;
    }
    
    // FIXED: Calculate CBR based on packet rate and airtime
    // CBR ≈ (packets/sec * packet_duration) / interval
    double beaconHz = 1.0 / g_beaconInterval;
    double packetDuration = 0.001; // ~1ms for 200 byte packet at 6 Mbps
    double numNeighbors = CountNeighbors(node, 300.0);
    
    // Each neighbor transmits beaconHz packets/sec
    // This node hears all of them
    double cbr = std::min(1.0, numNeighbors * beaconHz * packetDuration);
    
    g_lastChannelSampleTime[nodeId] = currentTime;
    
    return cbr;
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
double CalculateReward(double pdr, double throughput, double avgNeighbors, double cbr) {
    // CRITICAL FIX: Reward function should heavily penalize poor PDR
    
    double beaconHz = 1.0 / g_beaconInterval;
    
    // 1. PDR reward (MOST CRITICAL): Use exponential scaling to heavily penalize low PDR
    double pdrReward = 0.0;
    if (pdr < 0.3) {
        // Severe penalty for very low PDR
        pdrReward = -50.0 + (pdr / 0.3) * 30.0; // -50 to -20
    } else if (pdr < 0.6) {
        pdrReward = -20.0 + ((pdr - 0.3) / 0.3) * 30.0; // -20 to 10
    } else if (pdr < 0.8) {
        pdrReward = 10.0 + ((pdr - 0.6) / 0.2) * 30.0; // 10 to 40
    } else {
        pdrReward = 40.0 + ((pdr - 0.8) / 0.2) * 20.0; // 40 to 60
    }
    
    // 2. Throughput reward: Target 20-40 kbps (realistic for VANET)
    double targetThroughputMin = 20000.0; // 20 kbps
    double targetThroughputMax = 40000.0; // 40 kbps
    double throughputReward = 0.0;
    
    if (throughput < targetThroughputMin) {
        throughputReward = (throughput / targetThroughputMin) * 10.0; // 0-10 points
    } else if (throughput <= targetThroughputMax) {
        throughputReward = 10.0 + ((throughput - targetThroughputMin) / 
                          (targetThroughputMax - targetThroughputMin)) * 10.0; // 10-20 points
    } else {
        // Penalty for excessive throughput (causes congestion)
        double excess = (throughput - targetThroughputMax) / targetThroughputMax;
        throughputReward = std::max(0.0, 20.0 - excess * 30.0);
    }
    
    // 3. BeaconHz reward: Optimal 6-10 Hz for VANET
    double beaconReward = 0.0;
    double targetBeaconMin = 6.0;
    double targetBeaconMax = 10.0;
    
    if (beaconHz < targetBeaconMin) {
        beaconReward = (beaconHz / targetBeaconMin) * 5.0; // 0-5 points (too slow = bad)
    } else if (beaconHz <= targetBeaconMax) {
        beaconReward = 5.0 + ((beaconHz - targetBeaconMin) / 
                      (targetBeaconMax - targetBeaconMin)) * 10.0; // 5-15 points
    } else {
        // Penalty for excessive beacon rate (causes congestion)
        double excess = (beaconHz - targetBeaconMax) / targetBeaconMax;
        beaconReward = std::max(-10.0, 15.0 - excess * 25.0);
    }
    
    // 4. Connectivity reward: Target 8-15 neighbors (realistic for 300m range)
    double connectivityReward = 0.0;
    if (avgNeighbors < 5.0) {
        // Too few neighbors = network fragmentation
        connectivityReward = -10.0 + (avgNeighbors / 5.0) * 10.0; // -10 to 0
    } else if (avgNeighbors <= 15.0) {
        connectivityReward = (avgNeighbors / 15.0) * 10.0; // 0-10 points
    } else {
        // Too many neighbors is fine (up to a point)
        connectivityReward = 10.0;
    }
    
    // 5. Congestion penalty (CBR-based)
    double congestionPenalty = 0.0;
    
    if (cbr > 0.3) {
        // CBR above 0.3 indicates channel congestion
        congestionPenalty = (cbr - 0.3) / 0.7 * 20.0; // 0-20 penalty
    }
    
    // 6. Stability penalty
    double stabilityPenalty = 0.0;
    if (pdr < 0.4) stabilityPenalty += 10.0; // Severe PDR penalty
    if (avgNeighbors < 3.0) stabilityPenalty += 10.0; // Network fragmentation
    if (beaconHz < 2.0 || beaconHz > 15.0) stabilityPenalty += 5.0; // Extreme beacon rates
    
    // Total reward (can be negative!)
    double totalReward = pdrReward + throughputReward + beaconReward + 
                        connectivityReward - stabilityPenalty - congestionPenalty;
    
    // Don't clamp to 0 - allow negative rewards to punish bad behavior!
    
    // Debug output
    std::cout << "[Reward Breakdown]" << std::endl;
    std::cout << "  PDR:          " << std::fixed << std::setprecision(2) << pdrReward 
              << " (PDR: " << std::setprecision(3) << pdr << ")" << std::endl;
    std::cout << "  Throughput:   " << throughputReward 
              << " (Tput: " << (int)(throughput/1000) << " kbps)" << std::endl;
    std::cout << "  BeaconHz:     " << beaconReward << " (Hz: " << beaconHz << ")" << std::endl;
    std::cout << "  Connectivity: " << connectivityReward 
              << " (Neighbors: " << avgNeighbors << ")" << std::endl;
    std::cout << "  Congestion:   -" << congestionPenalty << " (CBR: " << cbr << ")" << std::endl;
    std::cout << "  Stability:    -" << stabilityPenalty << std::endl;
    std::cout << "  TOTAL:        " << totalReward << std::endl;
    
    return totalReward;
}

// Reset windowed metrics for next measurement interval
void ResetMetricsWindow() {
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        // Save current window to previous (for debugging/comparison)
        g_previousWindow[i] = g_currentWindow[i];
        
        // Reset current window
        g_currentWindow[i].packetsSent = 0;
        g_currentWindow[i].packetsReceived = 0;
        g_currentWindow[i].expectedReceptions = 0;
        g_currentWindow[i].windowStart = Simulator::Now();
    }
}

// Log metrics and interact with RL agent
void LogMetrics() {
    double currentTime = Simulator::Now().GetSeconds();
    
    std::cout << "========================================" << std::endl;
    std::cout << "Time: " << currentTime << "s | BeaconHz: " << (1.0/g_beaconInterval)
              << " | TxPower: " << g_txPower << " dBm" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Calculate WINDOWED metrics (last logging interval)
    uint64_t windowSent = 0, windowRecv = 0, windowExpected = 0;
    uint64_t totalSent = 0, totalRecv = 0, totalExpectedRecv = 0;
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        Ptr<Node> node = g_nodes.Get(i);
        uint32_t nodeId = node->GetId();
        
        // WINDOWED counts (last interval only)
        uint64_t sent_window = g_currentWindow[nodeId].packetsSent;
        uint64_t recv_window = g_currentWindow[nodeId].packetsReceived;
        uint64_t expected_window = g_currentWindow[nodeId].expectedReceptions;
        
        windowSent += sent_window;
        windowRecv += recv_window;
        windowExpected += expected_window;
        
        // Cumulative counts (for display)
        uint64_t sent = g_packetsSent[nodeId];
        uint64_t recv = g_packetsReceived[nodeId];
        uint64_t expectedRecv = g_expectedReceptions[nodeId];
        totalSent += sent;
        totalRecv += recv;
        totalExpectedRecv += expectedRecv;
        
        // PDR from WINDOWED data (more responsive!)
        double nodePDR_window = (expected_window > 0) ? (double)recv_window / expected_window : 0.0;
        
        // Throughput from WINDOWED data
        double windowDuration = g_loggingInterval;
        double throughput_window = (recv_window * 200 * 8) / windowDuration;
        
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
        Vector pos = mobility->GetPosition();
        Vector vel = mobility->GetVelocity();
        
        uint32_t neighbors = CountNeighbors(node, 300.0);
        double avgDist = GetAvgInterVehicleDistance(node);
        
        std::cout << "Node[" << std::setw(2) << nodeId << "]"
                  << " Sent:" << std::setw(5) << sent
                  << " Recv:" << std::setw(5) << recv
                  << " (W:" << std::setw(4) << recv_window << ")"  // Show window recv
                  << " PDR:" << std::fixed << std::setprecision(3) << std::setw(6) << nodePDR_window
                  << " Tput:" << std::setw(8) << (int)throughput_window << "bps"
                  << " | Pos(" << std::setw(4) << (int)pos.x << "," << std::setw(2) << (int)pos.y << ")"
                  << " Vel(" << std::setw(2) << (int)vel.x << ",0)"
                  << " Neigh:" << std::setw(2) << neighbors
                  << std::endl;
    }
    
    // WINDOWED global PDR (what RL agent should see!)
    double windowPDR = (windowExpected > 0) ? (double)windowRecv / windowExpected : 0.0;
    
    // CUMULATIVE global PDR (for reference)
    double cumulativePDR = (totalExpectedRecv > 0) ? (double)totalRecv / totalExpectedRecv : 0.0;
    
    // Calculate WINDOWED average metrics
    double avgThroughput_window = 0.0, avgNeighbors = 0.0;
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        Ptr<Node> node = g_nodes.Get(i);
        uint32_t nodeId = node->GetId();
        
        uint64_t recv_window = g_currentWindow[nodeId].packetsReceived;
        double throughput_window = (recv_window * 200 * 8) / g_loggingInterval;
        avgThroughput_window += throughput_window;
        avgNeighbors += CountNeighbors(node, 300.0);
    }
    avgThroughput_window /= g_nodes.GetN();
    avgNeighbors /= g_nodes.GetN();
    
    // NEW: Calculate average Channel Busy Ratio
    double avgCBR = GetAverageCBR();
    
    std::cout << "========================================" << std::endl;
    std::cout << "WINDOW Sent:" << windowSent << " Recv:" << windowRecv 
              << " Expected:" << windowExpected 
              << " PDR:" << std::fixed << std::setprecision(3) << windowPDR << std::endl;
    std::cout << "CUMULATIVE Sent:" << totalSent << " Recv:" << totalRecv 
              << " PDR:" << cumulativePDR << std::endl;
    std::cout << "AvgThroughput:" << (int)avgThroughput_window << "bps/node"
              << " AvgNeighbors:" << std::setprecision(1) << avgNeighbors 
              << " CBR:" << std::setprecision(3) << avgCBR << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    // RL Agent Interaction - USE WINDOWED METRICS!
    if (g_enableRL && g_rlInterface) {
        // Build state dictionary with WINDOWED metrics
        std::map<std::string, double> state;
        state["time"] = currentTime;
        state["PDR"] = windowPDR;  // WINDOWED PDR, not cumulative!
        state["throughput"] = avgThroughput_window;  // WINDOWED throughput!
        state["packetsSent"] = windowSent;  // WINDOWED count
        state["packetsReceived"] = windowRecv;  // WINDOWED count
        state["avgNeighbors"] = avgNeighbors;
        state["beaconHz"] = 1.0 / g_beaconInterval;
        state["txPower"] = g_txPower;
        state["numVehicles"] = g_nodes.GetN();
        state["CBR"] = avgCBR;  // Channel Busy Ratio
        
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
        
        // Calculate reward from WINDOWED metrics
        double reward = CalculateReward(windowPDR, avgThroughput_window, avgNeighbors, avgCBR);
        
        // For continuous learning, 'done' should always be false (except at end)
        bool done = (currentTime >= g_simulationTime - g_loggingInterval - 0.1);
        
        // Send reward to RL agent
        g_rlInterface->SendReward(reward, done);
        
        std::cout << ">>> RL Reward: " << std::fixed << std::setprecision(3) 
                  << reward << " | Done: " << (done ? "Yes" : "No") << " <<<" << std::endl;
        std::cout << std::endl;
    }
    
    // RESET WINDOW for next interval (CRITICAL for continuous learning!)
    ResetMetricsWindow();
    
    if (currentTime < g_simulationTime - 0.1) {
        Simulator::Schedule(Seconds(g_loggingInterval), &LogMetrics);
    }
}

// Update beacon interval
void UpdateBeaconInterval(double newInterval) {
    g_beaconInterval = newInterval;
    
    // Update data rate for all OnOff applications
    for (uint32_t i = 0; i < g_onoffApps.GetN(); i++) {
        Ptr<OnOffApplication> onoffApp = DynamicCast<OnOffApplication>(g_onoffApps.Get(i));
        if (onoffApp) {
            // Calculate new data rate based on new beacon interval
            // PacketSize = 200 bytes = 1600 bits
            // DataRate = bits per second = (1600 bits) / (beacon interval in seconds)
            DataRate newRate((200 * 8) / newInterval);
            onoffApp->SetAttribute("DataRate", DataRateValue(newRate));
        }
    }
    
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
    
    // Configure MAC layer to reduce collisions
    Config::SetDefault("ns3::WifiMacQueue::MaxSize", QueueSizeValue(QueueSize("400p")));
    Config::SetDefault("ns3::WifiMacQueue::MaxDelay", TimeValue(Seconds(0.5)));
    
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                  "DataMode", StringValue("OfdmRate6MbpsBW10MHz"));
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    
    g_devices = wifi.Install(wifiPhy, wifiMac, g_nodes);
    
    // Mobility - Realistic Highway Scenario
    MobilityHelper mobility;
    
    // Define highway parameters
    double highwayLength = 3000.0;  // 3 km circular highway
    double laneWidth = 3.5;         // Standard lane width (meters)
    uint32_t numLanes = 4;
    
    // Use GaussMarkovMobilityModel for realistic vehicle movement
    // Attributes use RandomVariableStream (StringValue), not direct double values
    mobility.SetMobilityModel("ns3::GaussMarkovMobilityModel",
                             "Bounds", BoxValue(Box(0, highwayLength, 0, numLanes * laneWidth, 0, 0)),
                             "TimeStep", TimeValue(Seconds(0.5)),
                             "Alpha", DoubleValue(0.85),  // Higher = smoother movement (memory effect)
                             "MeanVelocity", StringValue("ns3::UniformRandomVariable[Min=22.0|Max=33.0]"),  // 79-119 km/h
                             "MeanDirection", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=0.1]"),   // Mostly X-axis (0-5.7 degrees)
                             "MeanPitch", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"),
                             "NormalVelocity", StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=3.0|Bound=10.0]"),
                             "NormalDirection", StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=0.2|Bound=0.5]"),
                             "NormalPitch", StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=0.0|Bound=0.0]"));
    
    // Set initial positions along highway lanes
    Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
    
    for (uint32_t i = 0; i < g_numVehicles; i++) {
        uint32_t lane = i % numLanes;
        double x = ((i / numLanes) * 30.0);  // Spread vehicles every 30m
        // Wrap around if exceeds highway length
        while (x >= highwayLength) x -= highwayLength;
        
        double y = lane * laneWidth + laneWidth / 2.0;  // Center of lane
        posAlloc->Add(Vector(x, y, 0.0));
    }
    
    mobility.SetPositionAllocator(posAlloc);
    mobility.Install(g_nodes);
    
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
        
        // Sender - CRITICAL FIXES for collision reduction
        OnOffHelper onoff("ns3::UdpSocketFactory",
                         InetSocketAddress(Ipv4Address("10.1.255.255"), port));
        onoff.SetConstantRate(DataRate((200 * 8) / g_beaconInterval));
        onoff.SetAttribute("PacketSize", UintegerValue(200));
        
        // FIX 1: Use exponential distribution for OnTime to add randomness
        // This prevents synchronized periodic transmissions that cause collisions
        std::stringstream onTimeStream;
        onTimeStream << "ns3::ExponentialRandomVariable[Mean=" << g_beaconInterval << "]";
        onoff.SetAttribute("OnTime", StringValue(onTimeStream.str()));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
        
        ApplicationContainer app = onoff.Install(g_nodes.Get(i));
        
        // FIX 2: Randomize start times across full beacon interval
        // This spreads initial transmissions to avoid synchronized starts
        Ptr<UniformRandomVariable> startTimeRandom = CreateObject<UniformRandomVariable>();
        startTimeRandom->SetAttribute("Min", DoubleValue(1.0));
        startTimeRandom->SetAttribute("Max", DoubleValue(1.0 + g_beaconInterval));
        double startTime = startTimeRandom->GetValue();
        
        app.Start(Seconds(startTime));
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
        
        // NEW: Initialize windowed metrics for continuous learning
        g_currentWindow[i] = MetricsWindow();
        g_previousWindow[i] = MetricsWindow();
        g_currentWindow[i].windowStart = Seconds(0.0);
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
