OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(3.4615896988688237) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(3.722554177218983) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(3.046667673187682) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(3.6154031541086624) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(3.11866315959013) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(3.022236236615406) q[2];
cx q[3],q[2];
rx(2.150021949574629) q[0];
rx(2.150021949574629) q[1];
rx(2.150021949574629) q[2];
rx(2.150021949574629) q[3];
cx q[0],q[1];
rz(6.130235161484843) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(6.592385144656873) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(5.395437044903829) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(6.402628117798777) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.522936055649302) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(5.352170665999971) q[2];
cx q[3],q[2];
rx(3.0775860238193498) q[0];
rx(3.0775860238193498) q[1];
rx(3.0775860238193498) q[2];
rx(3.0775860238193498) q[3];
