OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
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
cx q[0],q[4];
rz(0.43058215964403745) q[4];
cx q[0],q[4];
cx q[0],q[8];
rz(0.3631149393023334) q[8];
cx q[0],q[8];
cx q[9],q[0];
rz(0.40228305059564584) q[0];
cx q[9],q[0];
cx q[1],q[2];
rz(0.4953874272852849) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(0.422213001721725) q[3];
cx q[1],q[3];
cx q[7],q[1];
rz(0.4387992835831834) q[1];
cx q[7],q[1];
cx q[2],q[4];
rz(0.365071111418169) q[4];
cx q[2],q[4];
cx q[11],q[2];
rz(0.4377941986382037) q[2];
cx q[11],q[2];
cx q[3],q[10];
rz(0.32163782660725976) q[10];
cx q[3],q[10];
cx q[11],q[3];
rz(0.5420537087866674) q[3];
cx q[11],q[3];
cx q[5],q[4];
rz(0.4483627524723946) q[4];
cx q[5],q[4];
cx q[5],q[6];
rz(0.45115072597567313) q[6];
cx q[5],q[6];
cx q[10],q[5];
rz(0.4732832288509039) q[5];
cx q[10],q[5];
cx q[6],q[7];
rz(0.46242146288554575) q[7];
cx q[6],q[7];
cx q[8],q[6];
rz(0.47789372678373504) q[6];
cx q[8],q[6];
cx q[9],q[7];
rz(0.44749721235631734) q[7];
cx q[9],q[7];
cx q[11],q[8];
rz(0.4245985749701109) q[8];
cx q[11],q[8];
cx q[10],q[9];
rz(0.4968276452153803) q[9];
cx q[10],q[9];
rx(2.2139847111325075) q[0];
rx(2.2139847111325075) q[1];
rx(2.2139847111325075) q[2];
rx(2.2139847111325075) q[3];
rx(2.2139847111325075) q[4];
rx(2.2139847111325075) q[5];
rx(2.2139847111325075) q[6];
rx(2.2139847111325075) q[7];
rx(2.2139847111325075) q[8];
rx(2.2139847111325075) q[9];
rx(2.2139847111325075) q[10];
rx(2.2139847111325075) q[11];
