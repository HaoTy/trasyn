OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
cx q[0],q[3];
rz(4.867334974917799) q[3];
cx q[0],q[3];
cx q[0],q[5];
rz(4.729958466291288) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(3.6801875098315144) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(4.438468665381447) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(5.030861018679841) q[3];
cx q[1],q[3];
cx q[4],q[1];
rz(4.429915091004772) q[1];
cx q[4],q[1];
cx q[2],q[4];
rz(4.9418849781116405) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(4.161774420359138) q[2];
cx q[5],q[2];
cx q[6],q[3];
rz(5.113267309794994) q[3];
cx q[6],q[3];
cx q[7],q[4];
rz(4.611925504930282) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(4.479425992666388) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(4.293198245318327) q[6];
cx q[7],q[6];
rx(2.7630381175553347) q[0];
rx(2.7630381175553347) q[1];
rx(2.7630381175553347) q[2];
rx(2.7630381175553347) q[3];
rx(2.7630381175553347) q[4];
rx(2.7630381175553347) q[5];
rx(2.7630381175553347) q[6];
rx(2.7630381175553347) q[7];
