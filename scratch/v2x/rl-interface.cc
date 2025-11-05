#include "rl-interface.h"
#include "ns3/log.h"
#include <iostream>
#include <nlohmann/json.hpp>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("RLInterface");

using json = nlohmann::json;

RLInterface::RLInterface()
  : m_ctx(1), m_socket(m_ctx, zmq::socket_type::req)
{}

RLInterface::RLInterface(const std::string& addr)
  : m_ctx(1), m_socket(m_ctx, zmq::socket_type::req)
{
  Init(addr);
}

RLInterface::~RLInterface()
{
  m_socket.close();
  m_ctx.close();
}

void RLInterface::Init(const std::string& addr)
{
  m_addr = addr;
  m_socket.connect(addr);
  NS_LOG_UNCOND("[RLInterface] Connected to " << addr);
}

void RLInterface::SendState(const std::vector<double>& state)
{
  json jmsg;
  jmsg["state"] = state;
  std::string data = jmsg.dump();

  zmq::message_t msg(data.begin(), data.end());
  m_socket.send(msg, zmq::send_flags::none);

  NS_LOG_UNCOND("[RLInterface] Sent state to agent");
}

int RLInterface::ReceiveAction()
{
  zmq::message_t reply;
  m_socket.recv(reply, zmq::recv_flags::none);
  std::string replyStr(static_cast<char*>(reply.data()), reply.size());

  json jreply = json::parse(replyStr);
  int action = jreply["action"];
  NS_LOG_UNCOND("[RLInterface] Received action: " << action);
  return action;
}

void RLInterface::SendReward(double reward, bool done)
{
  std::ostringstream oss;
  oss << reward << "," << (done ? 1 : 0);
  std::string data = oss.str();

  zmq::message_t msg(data.begin(), data.end());
  m_socket.send(msg, zmq::send_flags::none);

  // Wait for agent acknowledgment to maintain REQ/REP sync
  zmq::message_t ack;
  m_socket.recv(ack, zmq::recv_flags::none);
  std::string ackStr(static_cast<char*>(ack.data()), ack.size());
  NS_LOG_UNCOND("[RLInterface] Agent ACK: " << ackStr);
}

} // namespace ns3
