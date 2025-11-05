# CMakeLists.txt for scratch/v2x

# Collect all .cc files in this directory
file(GLOB V2X_SOURCES CONFIGURE_DEPENDS *.cc)

# Create executable for this scratch
add_executable(scratch_v2x_v2x ${V2X_SOURCES})

# Link with NS-3 core libraries and ZeroMQ
target_link_libraries(scratch_v2x_v2x
    PRIVATE
      ${ns3-libs}
      ${ns3-contrib-libs}
      zmq
)

# Set output directory for the binary
set_target_properties(scratch_v2x_v2x PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_OUTPUT_DIRECTORY}/scratch/v2x"
)