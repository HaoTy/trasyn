OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.870244444892499) q[1];
cx q[0],q[1];
h q[2];
cx q[0],q[2];
rz(2.0293818513130404) q[2];
cx q[0],q[2];
cx q[1],q[2];
rz(1.8632501870820442) q[2];
cx q[1],q[2];
h q[3];
cx q[3],q[0];
rz(-1.238526434934153) q[0];
h q[0];
rz(1.3394646675165873) q[0];
h q[0];
rz(3*pi) q[0];
cx q[3],q[0];
cx q[3],q[1];
rz(-1.321389738782977) q[1];
h q[1];
rz(1.3394646675165873) q[1];
h q[1];
rz(3*pi) q[1];
cx q[3],q[1];
cx q[0],q[1];
rz(8.167843651738657) q[1];
cx q[0],q[1];
cx q[3],q[2];
rz(-1.2748571531649766) q[2];
h q[2];
rz(1.3394646675165873) q[2];
h q[2];
rz(3*pi) q[2];
cx q[3],q[2];
h q[3];
rz(4.943720639662999) q[3];
h q[3];
cx q[0],q[2];
rz(8.328207523830894) q[2];
cx q[0],q[2];
cx q[1],q[2];
rz(1.8776101822814208) q[2];
cx q[1],q[2];
cx q[3],q[0];
rz(-4.3654522327503855) q[0];
h q[0];
rz(2.0045509576928113) q[0];
h q[0];
cx q[3],q[0];
cx q[3],q[1];
rz(-4.448954160791346) q[1];
h q[1];
rz(2.0045509576928113) q[1];
h q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(-4.402062950383867) q[2];
h q[2];
rz(2.0045509576928113) q[2];
h q[2];
cx q[3],q[2];
h q[3];
rz(2.0045509576928113) q[3];
h q[3];
