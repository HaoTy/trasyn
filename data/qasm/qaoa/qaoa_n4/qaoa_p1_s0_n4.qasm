OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(0.824852747925365) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(0.7767612114350563) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(0.7282366343241118) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(0.7964751994641986) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(0.927457949080795) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(1.0264226081081043) q[2];
cx q[3],q[2];
rx(3.1758633703072388) q[0];
rx(3.1758633703072388) q[1];
rx(3.1758633703072388) q[2];
rx(3.1758633703072388) q[3];
