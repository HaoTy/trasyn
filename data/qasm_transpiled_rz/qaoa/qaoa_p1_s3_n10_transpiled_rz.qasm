OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
h q[1];
cx q[0],q[1];
rz(5.564732814627146) q[1];
cx q[0],q[1];
h q[2];
cx q[1],q[2];
rz(6.506746708335364) q[2];
cx q[1],q[2];
h q[3];
cx q[3],q[1];
rz(-2.9295921951128983) q[1];
h q[1];
rz(0.9503747589884339) q[1];
h q[1];
rz(3*pi) q[1];
cx q[3],q[1];
h q[4];
cx q[2],q[4];
rz(5.584577009788797) q[4];
cx q[2],q[4];
h q[5];
cx q[3],q[5];
rz(7.213620643192028) q[5];
cx q[3],q[5];
cx q[4],q[5];
rz(6.039702801967945) q[5];
cx q[4],q[5];
h q[6];
cx q[6],q[3];
rz(-3.7458002920453928) q[3];
h q[3];
rz(0.9503747589884339) q[3];
h q[3];
rz(3*pi) q[3];
cx q[6],q[3];
cx q[6],q[4];
rz(-3.8473980816764315) q[4];
h q[4];
rz(0.9503747589884339) q[4];
h q[4];
rz(3*pi) q[4];
cx q[6],q[4];
h q[7];
cx q[0],q[7];
rz(5.663390600136976) q[7];
cx q[0],q[7];
cx q[7],q[2];
rz(-3.5735902753894613) q[2];
h q[2];
rz(0.9503747589884339) q[2];
h q[2];
rz(3*pi) q[2];
cx q[7],q[2];
h q[8];
cx q[8],q[0];
rz(-3.9578110433479785) q[0];
h q[0];
rz(0.9503747589884339) q[0];
h q[0];
rz(3*pi) q[0];
cx q[8],q[0];
cx q[8],q[7];
rz(-3.323337747734707) q[7];
h q[7];
rz(0.9503747589884339) q[7];
h q[7];
rz(3*pi) q[7];
cx q[8],q[7];
h q[9];
cx q[9],q[5];
rz(-3.765008403233519) q[5];
h q[5];
rz(0.9503747589884339) q[5];
h q[5];
rz(3*pi) q[5];
cx q[9],q[5];
cx q[9],q[6];
rz(-2.8988979477239396) q[6];
h q[6];
rz(0.9503747589884339) q[6];
h q[6];
rz(3*pi) q[6];
cx q[9],q[6];
cx q[9],q[8];
rz(-3.2965675898604023) q[8];
h q[8];
rz(0.9503747589884339) q[8];
h q[8];
rz(3*pi) q[8];
cx q[9],q[8];
h q[9];
rz(5.332810548191151) q[9];
h q[9];
