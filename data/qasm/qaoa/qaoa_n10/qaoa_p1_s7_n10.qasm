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
cx q[0],q[2];
rz(0.2319067278658195) q[2];
cx q[0],q[2];
cx q[0],q[4];
rz(0.2073646086325731) q[4];
cx q[0],q[4];
cx q[9],q[0];
rz(0.22085291790135045) q[0];
cx q[9],q[0];
cx q[1],q[2];
rz(0.19380573972619755) q[2];
cx q[1],q[2];
cx q[1],q[5];
rz(0.20941773200590436) q[5];
cx q[1],q[5];
cx q[8],q[1];
rz(0.22082762610261933) q[1];
cx q[8],q[1];
cx q[7],q[2];
rz(0.25471447958267907) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(0.21486283592056366) q[4];
cx q[3],q[4];
cx q[3],q[7];
rz(0.20214505886687278) q[7];
cx q[3],q[7];
cx q[8],q[3];
rz(0.2628502161281249) q[3];
cx q[8],q[3];
cx q[6],q[4];
rz(0.24462055809861102) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(0.1803522429867987) q[6];
cx q[5],q[6];
cx q[9],q[5];
rz(0.20023435725845745) q[5];
cx q[9],q[5];
cx q[9],q[6];
rz(0.23184956215421948) q[6];
cx q[9],q[6];
cx q[8],q[7];
rz(0.22439896333061532) q[7];
cx q[8],q[7];
rx(5.70850489703503) q[0];
rx(5.70850489703503) q[1];
rx(5.70850489703503) q[2];
rx(5.70850489703503) q[3];
rx(5.70850489703503) q[4];
rx(5.70850489703503) q[5];
rx(5.70850489703503) q[6];
rx(5.70850489703503) q[7];
rx(5.70850489703503) q[8];
rx(5.70850489703503) q[9];
