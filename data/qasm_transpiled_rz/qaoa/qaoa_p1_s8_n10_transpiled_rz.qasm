OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
h q[1];
h q[2];
cx q[1],q[2];
rz(6.12365083221874) q[2];
cx q[1],q[2];
h q[3];
cx q[0],q[3];
rz(5.7794413931787645) q[3];
cx q[0],q[3];
h q[4];
cx q[0],q[4];
rz(5.452074534903264) q[4];
cx q[0],q[4];
cx q[2],q[4];
rz(4.499424996051757) q[4];
cx q[2],q[4];
h q[5];
cx q[1],q[5];
rz(5.660524548048635) q[5];
cx q[1],q[5];
h q[6];
cx q[6],q[2];
rz(-3.0326087457539646) q[2];
h q[2];
rz(0.22052604401435572) q[2];
h q[2];
rz(3*pi) q[2];
cx q[6],q[2];
cx q[5],q[6];
rz(6.081607177400636) q[6];
cx q[5],q[6];
h q[7];
cx q[7],q[0];
rz(-2.9800493349127666) q[0];
h q[0];
rz(0.22052604401435572) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[7],q[1];
rz(-4.11194541595308) q[1];
h q[1];
rz(0.22052604401435572) q[1];
h q[1];
rz(3*pi) q[1];
cx q[7],q[1];
cx q[3],q[7];
rz(-4.587140500737981) q[7];
h q[7];
rz(0.22052604401435527) q[7];
h q[7];
rz(3*pi) q[7];
cx q[3],q[7];
h q[8];
cx q[8],q[4];
rz(-2.3244053119220793) q[4];
h q[4];
rz(0.22052604401435572) q[4];
h q[4];
rz(3*pi) q[4];
cx q[8],q[4];
cx q[8],q[5];
rz(-2.69356797993783) q[5];
h q[5];
rz(0.22052604401435572) q[5];
h q[5];
rz(3*pi) q[5];
cx q[8],q[5];
h q[9];
cx q[9],q[3];
rz(-3.504812769485825) q[3];
h q[3];
rz(0.22052604401435572) q[3];
h q[3];
rz(3*pi) q[3];
cx q[9],q[3];
cx q[9],q[6];
rz(-3.5609288892233395) q[6];
h q[6];
rz(0.22052604401435572) q[6];
h q[6];
rz(3*pi) q[6];
cx q[9],q[6];
cx q[9],q[8];
rz(-2.424751560162243) q[8];
h q[8];
rz(0.22052604401435527) q[8];
h q[8];
rz(3*pi) q[8];
cx q[9],q[8];
h q[9];
rz(6.06265926316523) q[9];
h q[9];
