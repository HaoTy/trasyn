OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
h q[0];
h q[1];
h q[2];
cx q[0],q[2];
rz(0.06675508983765259) q[2];
cx q[0],q[2];
cx q[1],q[2];
rz(0.07293087717410962) q[2];
cx q[1],q[2];
h q[3];
cx q[1],q[3];
rz(0.0776757138680224) q[3];
cx q[1],q[3];
h q[4];
cx q[0],q[4];
rz(0.07308164422350213) q[4];
cx q[0],q[4];
h q[5];
h q[6];
cx q[6],q[1];
rz(0.075288654946986) q[1];
h q[1];
rz(0.9591739667200514) q[1];
h q[1];
cx q[6],q[1];
h q[7];
cx q[4],q[7];
rz(0.06508449112210163) q[7];
cx q[4],q[7];
h q[8];
cx q[6],q[8];
rz(0.07853120135966847) q[8];
cx q[6],q[8];
h q[9];
cx q[9],q[2];
rz(0.07402576669445704) q[2];
h q[2];
rz(0.9591739667200514) q[2];
h q[2];
cx q[9],q[2];
cx q[3],q[9];
rz(0.06562047491204843) q[9];
cx q[3],q[9];
cx q[9],q[4];
rz(0.06581920673617692) q[4];
h q[4];
rz(0.9591739667200514) q[4];
h q[4];
cx q[9],q[4];
h q[9];
rz(0.9591739667200514) q[9];
h q[9];
h q[10];
cx q[8],q[10];
rz(0.07169884116699711) q[10];
cx q[8],q[10];
h q[11];
cx q[5],q[11];
rz(0.07232036771032069) q[11];
cx q[5],q[11];
cx q[11],q[8];
rz(0.07507419530509818) q[8];
h q[8];
rz(0.9591739667200514) q[8];
h q[8];
cx q[11],q[8];
cx q[10],q[11];
rz(0.066217897999028) q[11];
h q[11];
rz(0.9591739667200514) q[11];
h q[11];
cx q[10],q[11];
h q[12];
cx q[7],q[12];
rz(0.0763344114510646) q[12];
cx q[7],q[12];
h q[13];
cx q[13],q[3];
rz(0.07667246198070288) q[3];
h q[3];
rz(0.9591739667200514) q[3];
h q[3];
cx q[13],q[3];
cx q[5],q[13];
rz(0.06761898983634924) q[13];
cx q[5],q[13];
cx q[13],q[7];
rz(0.07811160334369838) q[7];
h q[7];
rz(0.9591739667200514) q[7];
h q[7];
cx q[13],q[7];
h q[13];
rz(0.9591739667200514) q[13];
h q[13];
h q[14];
cx q[14],q[5];
rz(0.06325044103055255) q[5];
h q[5];
rz(0.9591739667200514) q[5];
h q[5];
cx q[14],q[5];
cx q[14],q[6];
rz(0.07399791502726671) q[6];
h q[6];
rz(0.9591739667200514) q[6];
h q[6];
cx q[14],q[6];
cx q[12],q[14];
rz(0.059086397818385095) q[14];
h q[14];
rz(0.9591739667200514) q[14];
h q[14];
cx q[12],q[14];
h q[15];
cx q[15],q[0];
rz(0.06708239570731234) q[0];
h q[0];
rz(0.9591739667200514) q[0];
h q[0];
cx q[15],q[0];
cx q[15],q[10];
rz(0.07743777651908879) q[10];
h q[10];
rz(0.9591739667200514) q[10];
h q[10];
cx q[15],q[10];
cx q[15],q[12];
rz(0.07381401147517153) q[12];
h q[12];
rz(0.9591739667200514) q[12];
h q[12];
cx q[15],q[12];
h q[15];
rz(0.9591739667200514) q[15];
h q[15];
