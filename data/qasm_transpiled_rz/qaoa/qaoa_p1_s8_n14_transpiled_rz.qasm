OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[1],q[3];
rz(0.6925897638805525) q[3];
cx q[1],q[3];
h q[4];
cx q[1],q[4];
rz(0.7304038848452787) q[4];
cx q[1],q[4];
h q[5];
cx q[2],q[5];
rz(0.5664102750471968) q[5];
cx q[2],q[5];
h q[6];
cx q[0],q[6];
rz(0.7404656437687975) q[6];
cx q[0],q[6];
h q[7];
cx q[0],q[7];
rz(0.6944193351516744) q[7];
cx q[0],q[7];
cx q[7],q[1];
rz(0.645446087296957) q[1];
h q[1];
rz(2.499742484768367) q[1];
h q[1];
cx q[7],q[1];
cx q[3],q[7];
rz(0.717552896177704) q[7];
h q[7];
rz(2.499742484768367) q[7];
h q[7];
cx q[3],q[7];
h q[8];
cx q[6],q[8];
rz(0.6401534169727118) q[8];
cx q[6],q[8];
h q[9];
cx q[9],q[0];
rz(0.7702718093100636) q[0];
h q[0];
rz(2.499742484768367) q[0];
h q[0];
cx q[9],q[0];
h q[10];
cx q[4],q[10];
rz(0.6937398402166605) q[10];
cx q[4],q[10];
cx q[5],q[10];
rz(0.7561774171294964) q[10];
cx q[5],q[10];
cx q[8],q[10];
rz(0.6833356900442062) q[10];
h q[10];
rz(2.499742484768367) q[10];
h q[10];
cx q[8],q[10];
h q[11];
cx q[11],q[3];
rz(0.6478049015163236) q[3];
h q[3];
rz(2.499742484768367) q[3];
h q[3];
cx q[11],q[3];
cx q[11],q[6];
rz(0.5517702816367773) q[6];
h q[6];
rz(2.499742484768367) q[6];
h q[6];
cx q[11],q[6];
cx q[9],q[11];
rz(0.7610448626820405) q[11];
h q[11];
rz(2.499742484768367) q[11];
h q[11];
cx q[9],q[11];
h q[12];
cx q[2],q[12];
rz(0.6662974183635921) q[12];
cx q[2],q[12];
cx q[12],q[4];
rz(0.7160766539146461) q[4];
h q[4];
rz(2.499742484768367) q[4];
h q[4];
cx q[12],q[4];
cx q[12],q[8];
rz(0.7020494316477759) q[8];
h q[8];
rz(2.499742484768367) q[8];
h q[8];
cx q[12],q[8];
h q[12];
rz(2.499742484768367) q[12];
h q[12];
h q[13];
cx q[13],q[2];
rz(0.6548999276772811) q[2];
h q[2];
rz(2.499742484768367) q[2];
h q[2];
cx q[13],q[2];
cx q[13],q[5];
rz(0.7105929451436221) q[5];
h q[5];
rz(2.499742484768367) q[5];
h q[5];
cx q[13],q[5];
cx q[13],q[9];
rz(0.6688154328856246) q[9];
h q[9];
rz(2.499742484768367) q[9];
h q[9];
cx q[13],q[9];
h q[13];
rz(2.499742484768367) q[13];
h q[13];
