/*
 * VANET Scenario for Reinforcement Learning
 * Compatible with NS-3 3.40
 * 
 * Simple working version that logs metrics
 * Beacon frequency and TX power can be dynamically updated
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
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

std::map<uint32_t, uint64_t> g_packetsSent;
std::map<uint32_t, uint64_t> g_packetsReceived;
NodeContainer g_nodes;
NetDeviceContainer g_devices;

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

// Log metrics
void LogMetrics() {
    double currentTime = Simulator::Now().GetSeconds();
    
    std::cout << "========================================" << std::endl;
    std::cout << "Time: " << currentTime << "s | BeaconHz: " << (1.0/g_beaconInterval)
              << " | TxPower: " << g_txPower << " dBm" << std::endl;
    std::cout << "========================================" << std::endl;
    
    uint64_t totalSent = 0, totalRecv = 0;
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        Ptr<Node> node = g_nodes.Get(i);
        uint32_t nodeId = node->GetId();
        
        uint64_t sent = g_packetsSent[nodeId];
        uint64_t recv = g_packetsReceived[nodeId];
        totalSent += sent;
        totalRecv += recv;
        
        // Calculate PDR for this node
        // In broadcast: node should receive (N-1) packets for each packet sent by others
        // Expected packets = sum of all other nodes' sent packets
        uint64_t expectedRecv = 0;
        for (uint32_t j = 0; j < g_nodes.GetN(); j++) {
            if (i != j) {
                expectedRecv += g_packetsSent[j];
            }
        }
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
                  << " PDR:" << std::fixed << std::setprecision(3) << std::setw(6) << nodePDR
                  << " Tput:" << std::setw(8) << (int)throughput << "bps"
                  << " | Pos(" << std::setw(4) << (int)pos.x << "," << std::setw(2) << (int)pos.y << ")"
                  << " Vel(" << std::setw(2) << (int)vel.x << ",0)"
                  << " Neigh:" << std::setw(2) << neighbors
                  << " Dist:" << std::setw(4) << (int)avgDist << "m"
                  << std::endl;
    }
    
    // Global PDR: total received vs total that should have been received
    uint64_t totalExpectedRecv = totalSent * (g_nodes.GetN() - 1);
    double globalPDR = (totalExpectedRecv > 0) ? (double)totalRecv / totalExpectedRecv : 0.0;
    
    std::cout << "========================================" << std::endl;
    std::cout << "TOTAL Sent:" << totalSent << " Recv:" << totalRecv 
              << " GlobalPDR:" << std::fixed << std::setprecision(3) << globalPDR << std::endl;
    std::cout << std::endl;
    
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
    std::cout << "=========================================\n\n";
    
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
        double speed = 25.0 + (rand() % 10);
        mob->SetVelocity(Vector(speed, 0.0, 0.0));
    }
    
    // Internet stack
    InternetStackHelper internet;
    internet.Install(g_nodes);
    
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
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
                         InetSocketAddress(Ipv4Address("10.1.1.255"), port));
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
    }
    
    // Schedule logging
    Simulator::Schedule(Seconds(g_loggingInterval), &LogMetrics);
    
    // Dynamic parameter changes (for RL testing)
    Simulator::Schedule(Seconds(20.0), &UpdateBeaconInterval, 0.2);  // 5 Hz
    Simulator::Schedule(Seconds(40.0), &UpdateTxPower, 26.0);
    Simulator::Schedule(Seconds(60.0), &UpdateBeaconInterval, 0.1); // 10 Hz
    
    std::cout << "Starting simulation...\n\n";
    
    Simulator::Stop(Seconds(g_simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    std::cout << "\n=== Simulation Complete! ===\n";
    
    return 0;
}
