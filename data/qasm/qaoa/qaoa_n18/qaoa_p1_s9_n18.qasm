OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
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
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
cx q[0],q[2];
rz(6.542783715530756) q[2];
cx q[0],q[2];
cx q[0],q[4];
rz(5.93100324847652) q[4];
cx q[0],q[4];
cx q[17],q[0];
rz(5.3199487164477715) q[0];
cx q[17],q[0];
cx q[1],q[13];
rz(7.524576998773982) q[13];
cx q[1],q[13];
cx q[1],q[14];
rz(6.421556646087114) q[14];
cx q[1],q[14];
cx q[16],q[1];
rz(6.621500316245716) q[1];
cx q[16],q[1];
cx q[2],q[6];
rz(5.528716463925638) q[6];
cx q[2],q[6];
cx q[8],q[2];
rz(4.903903226826835) q[2];
cx q[8],q[2];
cx q[3],q[10];
rz(6.299893628534978) q[10];
cx q[3],q[10];
cx q[3],q[12];
rz(5.807132566718394) q[12];
cx q[3],q[12];
cx q[13],q[3];
rz(5.836814782565416) q[3];
cx q[13],q[3];
cx q[4],q[10];
rz(5.8956284554259115) q[10];
cx q[4],q[10];
cx q[15],q[4];
rz(5.146863577584365) q[4];
cx q[15],q[4];
cx q[5],q[8];
rz(5.213182636609521) q[8];
cx q[5],q[8];
cx q[5],q[14];
rz(5.824900093039103) q[14];
cx q[5],q[14];
cx q[17],q[5];
rz(6.854295200731454) q[5];
cx q[17],q[5];
cx q[6],q[9];
rz(5.557298233473647) q[9];
cx q[6],q[9];
cx q[11],q[6];
rz(5.804246750852339) q[6];
cx q[11],q[6];
cx q[7],q[8];
rz(4.641603280041413) q[8];
cx q[7],q[8];
cx q[7],q[11];
rz(6.628505752988733) q[11];
cx q[7],q[11];
cx q[12],q[7];
rz(5.216663847617668) q[7];
cx q[12],q[7];
cx q[9],q[12];
rz(5.761879054589719) q[12];
cx q[9],q[12];
cx q[15],q[9];
rz(4.970769239846952) q[9];
cx q[15],q[9];
cx q[13],q[10];
rz(5.007385775225653) q[10];
cx q[13],q[10];
cx q[16],q[11];
rz(5.423391524969569) q[11];
cx q[16],q[11];
cx q[15],q[14];
rz(5.042675780487686) q[14];
cx q[15],q[14];
cx q[17],q[16];
rz(6.01822474144373) q[16];
cx q[17],q[16];
rx(4.495757248158194) q[0];
rx(4.495757248158194) q[1];
rx(4.495757248158194) q[2];
rx(4.495757248158194) q[3];
rx(4.495757248158194) q[4];
rx(4.495757248158194) q[5];
rx(4.495757248158194) q[6];
rx(4.495757248158194) q[7];
rx(4.495757248158194) q[8];
rx(4.495757248158194) q[9];
rx(4.495757248158194) q[10];
rx(4.495757248158194) q[11];
rx(4.495757248158194) q[12];
rx(4.495757248158194) q[13];
rx(4.495757248158194) q[14];
rx(4.495757248158194) q[15];
rx(4.495757248158194) q[16];
rx(4.495757248158194) q[17];
