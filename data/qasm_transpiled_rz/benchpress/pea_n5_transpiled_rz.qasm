OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(-0.6989526135777275) q[0];
h q[0];
rz(pi) q[0];
h q[0];
rz(4.209785907656325) q[0];
rz(-3*pi/2) q[1];
h q[1];
rz(pi/2) q[1];
h q[1];
rz(-pi/2) q[2];
h q[2];
rz(pi/2) q[2];
h q[2];
rz(4.251273076287474) q[2];
h q[3];
rz(-3*pi/8) q[4];
cx q[3],q[4];
rz(3*pi/8) q[4];
cx q[3],q[4];
rz(-7*pi/8) q[4];
h q[4];
rz(2.5802058126763816) q[4];
h q[4];
cx q[2],q[4];
rz(-3.7899297541797967) q[4];
h q[2];
rz(pi) q[2];
h q[2];
rz(18.388440017441546) q[2];
h q[4];
rz(1.7031757544375923) q[4];
h q[4];
rz(9.814313190905827) q[4];
cx q[2],q[4];
rz(0) q[4];
h q[4];
rz(2.132183167708309) q[4];
h q[4];
rz(6.507196629663621) q[4];
h q[4];
rz(pi/2) q[4];
h q[4];
rz(5*pi/2) q[4];
cx q[1],q[4];
rz(-3*pi/2) q[4];
h q[4];
rz(pi/2) q[4];
h q[4];
rz(12.735058373573864) q[4];
cx q[0],q[1];
rz(pi/4) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(-pi/4) q[1];
h q[1];
rz(-3*pi/8) q[1];
rz(pi/8) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(-pi/8) q[2];
cx q[1],q[2];
rz(pi/4) q[2];
cx q[1],q[2];
rz(-pi/4) q[2];
h q[2];
rz(-pi/4) q[2];
rz(pi/16) q[3];
cx q[0],q[3];
rz(-pi/16) q[3];
cx q[1],q[3];
rz(pi/8) q[3];
cx q[1],q[3];
rz(-pi/8) q[3];
cx q[2],q[3];
rz(pi/4) q[3];
cx q[2],q[3];
rz(-pi/4) q[3];
h q[3];
