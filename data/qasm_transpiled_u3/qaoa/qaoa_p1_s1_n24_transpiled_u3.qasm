OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,3.221213190352981) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cx q[0],q[3];
u3(0,0,3.4191460619154728) q[3];
cx q[0],q[3];
cx q[1],q[3];
u3(0,0,3.460639087780824) q[3];
cx q[1],q[3];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cx q[2],q[6];
u3(0,0,3.18984115051955) q[6];
cx q[2],q[6];
cx q[6],q[3];
u3(0.8930568766808797,pi/2,2.0344835653224997) q[3];
cx q[6],q[3];
u3(pi/2,0,pi) q[7];
cx q[7],q[1];
u3(0.8930568766808797,pi/2,1.618349534805735) q[1];
cx q[7],q[1];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
u3(pi/2,0,pi) q[10];
u3(pi/2,0,pi) q[11];
cx q[4],q[11];
u3(0,0,3.386814296645477) q[11];
cx q[4],q[11];
cx q[7],q[11];
u3(0,0,3.0737754992951434) q[11];
cx q[7],q[11];
u3(pi/2,0,pi) q[12];
cx q[12],q[0];
u3(0.8930568766808797,pi/2,1.9029522448147578) q[0];
cx q[12],q[0];
cx q[12],q[7];
u3(0.8930568766808797,pi/2,1.5219213495538293) q[7];
cx q[12],q[7];
u3(pi/2,0,pi) q[13];
cx q[8],q[13];
u3(0,0,3.073954998594908) q[13];
cx q[8],q[13];
cx q[10],q[13];
u3(0,0,2.6982110934412007) q[13];
cx q[10],q[13];
u3(pi/2,0,pi) q[14];
cx q[14],q[13];
u3(0.8930568766808797,pi/2,1.5470306486547418) q[13];
cx q[14],q[13];
u3(pi/2,0,pi) q[15];
cx q[15],q[6];
u3(0.8930568766808797,pi/2,1.5275961648589362) q[6];
cx q[15],q[6];
cx q[8],q[15];
u3(0,0,3.3281191162659316) q[15];
cx q[8],q[15];
cx q[9],q[15];
u3(0.8930568766808797,pi/2,1.2516544681616022) q[15];
cx q[9],q[15];
u3(pi/2,0,pi) q[16];
cx q[4],q[16];
u3(0,0,3.1633628419700277) q[16];
cx q[4],q[16];
cx q[9],q[16];
u3(0,0,2.9485497146953272) q[16];
cx q[9],q[16];
cx q[14],q[16];
u3(0.8930568766808797,pi/2,1.7089486908125604) q[16];
cx q[14],q[16];
u3(pi/2,0,pi) q[17];
cx q[17],q[8];
u3(0.8930568766808797,pi/2,1.9855844018463848) q[8];
cx q[17],q[8];
cx q[17],q[9];
u3(0.8930568766808797,pi/2,1.3411232592575253) q[9];
cx q[17],q[9];
u3(pi/2,0,pi) q[18];
cx q[5],q[18];
u3(0,0,2.715696537178162) q[18];
cx q[5],q[18];
cx q[10],q[18];
u3(0,0,3.1632038632613826) q[18];
cx q[10],q[18];
u3(pi/2,0,pi) q[19];
cx q[19],q[11];
u3(0.8930568766808797,pi/2,1.8395884661811266) q[11];
cx q[19],q[11];
cx q[19],q[17];
u3(0.8930568766808797,pi/2,1.5502939274158365) q[17];
cx q[19],q[17];
u3(pi/2,0,pi) q[20];
cx q[2],q[20];
u3(0,0,3.2580890009937633) q[20];
cx q[2],q[20];
cx q[20],q[4];
u3(0.8930568766808797,pi/2,1.5677101528544632) q[4];
cx q[20],q[4];
cx q[20],q[14];
u3(0.8930568766808797,pi/2,2.007941543131941) q[14];
cx q[20],q[14];
u3(5.390128430498707,-pi/2,pi/2) q[20];
u3(pi/2,0,pi) q[21];
cx q[5],q[21];
u3(0,0,3.1595905876178816) q[21];
cx q[5],q[21];
cx q[21],q[10];
u3(0.8930568766808797,pi/2,1.720041969157501) q[10];
cx q[21],q[10];
cx q[21],q[18];
u3(0.8930568766808797,pi/2,1.660008444045748) q[18];
cx q[21],q[18];
u3(5.390128430498707,-pi/2,pi/2) q[21];
u3(pi/2,0,pi) q[22];
cx q[22],q[5];
u3(0.8930568766808799,pi/2,1.2427166035610622) q[5];
cx q[22],q[5];
cx q[22],q[12];
u3(0.8930568766808797,pi/2,1.8203558060135965) q[12];
cx q[22],q[12];
u3(pi/2,0,pi) q[23];
cx q[23],q[2];
u3(0.8930568766808799,pi/2,1.4026021672341837) q[2];
cx q[23],q[2];
cx q[23],q[19];
u3(0.8930568766808797,pi/2,1.7502819516795824) q[19];
cx q[23],q[19];
cx q[23],q[22];
u3(0.8930568766808799,pi/2,1.0206359024852976) q[22];
cx q[23],q[22];
u3(5.390128430498707,-pi/2,pi/2) q[23];
