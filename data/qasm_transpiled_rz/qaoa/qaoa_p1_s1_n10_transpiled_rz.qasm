OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
h q[1];
h q[2];
cx q[1],q[2];
rz(3.5072695491705552) q[2];
cx q[1],q[2];
h q[3];
h q[4];
cx q[0],q[4];
rz(4.528670596042866) q[4];
cx q[0],q[4];
cx q[3],q[4];
rz(5.149141747290132) q[4];
cx q[3],q[4];
h q[5];
cx q[3],q[5];
rz(4.456865988701622) q[5];
cx q[3],q[5];
h q[6];
cx q[0],q[6];
rz(4.390042481273028) q[6];
cx q[0],q[6];
cx q[1],q[6];
rz(4.685733824877736) q[6];
cx q[1],q[6];
cx q[2],q[6];
rz(-2.5756217427820696) q[6];
h q[6];
rz(0.9022568807834954) q[6];
h q[6];
cx q[2],q[6];
h q[7];
cx q[7],q[1];
rz(-2.7003683888725085) q[1];
h q[1];
rz(0.9022568807834954) q[1];
h q[1];
cx q[7],q[1];
cx q[5],q[7];
rz(3.9475107199428314) q[7];
cx q[5],q[7];
h q[8];
cx q[8],q[0];
rz(-1.836787435587195) q[0];
h q[0];
rz(0.9022568807834954) q[0];
h q[0];
cx q[8],q[0];
cx q[8],q[2];
rz(-2.5311068216356487) q[2];
h q[2];
rz(0.9022568807834954) q[2];
h q[2];
cx q[8],q[2];
cx q[8],q[4];
rz(-1.9082673367236094) q[4];
h q[4];
rz(0.9022568807834954) q[4];
h q[4];
cx q[8],q[4];
h q[8];
rz(0.9022568807834954) q[8];
h q[8];
h q[9];
cx q[9],q[3];
rz(-1.8406824772910135) q[3];
h q[3];
rz(0.9022568807834954) q[3];
h q[3];
cx q[9],q[3];
cx q[9],q[5];
rz(-2.244573081819598) q[5];
h q[5];
rz(0.9022568807834954) q[5];
h q[5];
cx q[9],q[5];
cx q[9],q[7];
rz(-2.2606248244623517) q[7];
h q[7];
rz(0.9022568807834954) q[7];
h q[7];
cx q[9],q[7];
h q[9];
rz(0.9022568807834954) q[9];
h q[9];
