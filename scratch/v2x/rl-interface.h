#ifndef RL_INTERFACE_H
#define RL_INTERFACE_H

#include "ns3/object.h"
#include <zmq.hpp>
#include <string>
#include <vector>

namespace ns3 {

class RLInterface : public Object {
public:
  static TypeId GetTypeId(void) {
    static TypeId tid = TypeId("ns3::RLInterface")
      .SetParent<Object>()
      .SetGroupName("Core");
    return tid;
  }

  RLInterface();
  RLInterface(const std::string& addr);
  ~RLInterface();

  void Init(const std::string& addr);
  void SendState(const std::vector<double>& state);
  int ReceiveAction();
  void SendReward(double reward, bool done);

private:
  std::string m_addr;
  zmq::context_t m_ctx;
  zmq::socket_t m_socket;
};

} // namespace ns3

#endif // RL_INTERFACE_H
