OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(1.6304146945333515) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.5226806878489494) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(1.6391148854384985) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(1.4130352219913695) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(1.5020702328002467) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(1.7944463563867556) q[2];
cx q[3],q[2];
rx(1.7191865124640209) q[0];
rx(1.7191865124640209) q[1];
rx(1.7191865124640209) q[2];
rx(1.7191865124640209) q[3];
