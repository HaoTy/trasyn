OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
cx q[0],q[1];
rz(3.737238801372437) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(3.6856449760174637) q[4];
cx q[0],q[4];
cx q[9],q[0];
rz(4.335974771670441) q[0];
cx q[9],q[0];
cx q[1],q[2];
rz(3.999102702370842) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(3.622702944871802) q[1];
cx q[3],q[1];
cx q[2],q[5];
rz(3.8276735296929747) q[5];
cx q[2],q[5];
cx q[7],q[2];
rz(3.8300407603118187) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(3.629606229397892) q[4];
cx q[3],q[4];
cx q[8],q[3];
rz(4.312340022513904) q[3];
cx q[8],q[3];
cx q[6],q[4];
rz(3.2720606439381488) q[4];
cx q[6],q[4];
cx q[5],q[8];
rz(4.337007696280398) q[8];
cx q[5],q[8];
cx q[9],q[5];
rz(4.191123691427042) q[5];
cx q[9],q[5];
cx q[6],q[7];
rz(3.7742317246585153) q[7];
cx q[6],q[7];
cx q[9],q[6];
rz(3.80142699292588) q[6];
cx q[9],q[6];
cx q[8],q[7];
rz(4.194700055535812) q[7];
cx q[8],q[7];
rx(5.0713657526378) q[0];
rx(5.0713657526378) q[1];
rx(5.0713657526378) q[2];
rx(5.0713657526378) q[3];
rx(5.0713657526378) q[4];
rx(5.0713657526378) q[5];
rx(5.0713657526378) q[6];
rx(5.0713657526378) q[7];
rx(5.0713657526378) q[8];
rx(5.0713657526378) q[9];
