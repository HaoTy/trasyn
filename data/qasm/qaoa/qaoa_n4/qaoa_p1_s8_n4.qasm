OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(4.4203815973641225) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(5.377253305495358) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(5.48540286299165) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(5.893402008967591) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.273841866151566) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(4.152528920087036) q[2];
cx q[3],q[2];
rx(4.643890236024614) q[0];
rx(4.643890236024614) q[1];
rx(4.643890236024614) q[2];
rx(4.643890236024614) q[3];
