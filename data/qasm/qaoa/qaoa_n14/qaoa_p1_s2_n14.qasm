OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
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
cx q[0],q[1];
rz(1.1291546704093827) q[1];
cx q[0],q[1];
cx q[0],q[6];
rz(1.4069396309527473) q[6];
cx q[0],q[6];
cx q[12],q[0];
rz(1.3924482865051442) q[0];
cx q[12],q[0];
cx q[1],q[6];
rz(1.289982961354454) q[6];
cx q[1],q[6];
cx q[8],q[1];
rz(1.1553883707712091) q[1];
cx q[8],q[1];
cx q[2],q[8];
rz(1.3785012270643202) q[8];
cx q[2],q[8];
cx q[2],q[9];
rz(1.3774414488243203) q[9];
cx q[2],q[9];
cx q[13],q[2];
rz(1.5465315552557768) q[2];
cx q[13],q[2];
cx q[3],q[4];
rz(1.5800332073194956) q[4];
cx q[3],q[4];
cx q[3],q[5];
rz(1.4105202929397942) q[5];
cx q[3],q[5];
cx q[11],q[3];
rz(1.1824662016227745) q[3];
cx q[11],q[3];
cx q[4],q[7];
rz(1.29044745375731) q[7];
cx q[4],q[7];
cx q[11],q[4];
rz(1.4915243827349043) q[4];
cx q[11],q[4];
cx q[5],q[7];
rz(1.3229829077453295) q[7];
cx q[5],q[7];
cx q[10],q[5];
rz(1.3156593584054281) q[5];
cx q[10],q[5];
cx q[13],q[6];
rz(1.350247765658038) q[6];
cx q[13],q[6];
cx q[10],q[7];
rz(1.1195285893297997) q[7];
cx q[10],q[7];
cx q[10],q[8];
rz(1.265354893714972) q[8];
cx q[10],q[8];
cx q[9],q[12];
rz(1.371450413376309) q[12];
cx q[9],q[12];
cx q[13],q[9];
rz(1.3615980670857433) q[9];
cx q[13],q[9];
cx q[12],q[11];
rz(1.295697584627499) q[11];
cx q[12],q[11];
rx(1.3607933183368337) q[0];
rx(1.3607933183368337) q[1];
rx(1.3607933183368337) q[2];
rx(1.3607933183368337) q[3];
rx(1.3607933183368337) q[4];
rx(1.3607933183368337) q[5];
rx(1.3607933183368337) q[6];
rx(1.3607933183368337) q[7];
rx(1.3607933183368337) q[8];
rx(1.3607933183368337) q[9];
rx(1.3607933183368337) q[10];
rx(1.3607933183368337) q[11];
rx(1.3607933183368337) q[12];
rx(1.3607933183368337) q[13];
