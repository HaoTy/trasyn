OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rx(0.0665088227691338) q[3];
rx(0.0344275) q[2];
rx(-0.22422519168561286) q[1];
rx(-0.1160675) q[0];
h q[2];
cx q[3],q[2];
rx(-0.017820985350534065) q[3];
rz(-0.0344275) q[2];
h q[0];
cx q[1],q[0];
rx(0.060080959034873656) q[1];
rz(0.1160675) q[0];
cx q[1],q[0];
cx q[3],q[2];
h q[0];
h q[1];
h q[2];
h q[3];
h q[1];
h q[3];
