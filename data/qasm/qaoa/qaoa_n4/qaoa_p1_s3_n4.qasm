OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(1.6256175927813064) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.4528152184934109) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(1.6893515959264618) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(1.4634747865366136) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(1.3190811567252383) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(1.4520105671492862) q[2];
cx q[3],q[2];
rx(0.7704727397600557) q[0];
rx(0.7704727397600557) q[1];
rx(0.7704727397600557) q[2];
rx(0.7704727397600557) q[3];
