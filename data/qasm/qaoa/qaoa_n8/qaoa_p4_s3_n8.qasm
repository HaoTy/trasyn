OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
cx q[0],q[2];
rz(3.222322243817067) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(3.5750271837592478) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(2.6531050182396263) q[0];
cx q[4],q[0];
cx q[1],q[3];
rz(3.3781349400033713) q[3];
cx q[1],q[3];
cx q[1],q[5];
rz(3.515744864621031) q[5];
cx q[1],q[5];
cx q[7],q[1];
rz(3.3971712022868243) q[1];
cx q[7],q[1];
cx q[2],q[5];
rz(3.518395442506639) q[5];
cx q[2],q[5];
cx q[6],q[2];
rz(3.769468399882295) q[2];
cx q[6],q[2];
cx q[6],q[3];
rz(3.2449341441924315) q[3];
cx q[6],q[3];
cx q[4],q[6];
rz(3.5747490273912756) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(3.696391225726229) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(3.4822390550980034) q[5];
cx q[7],q[5];
rx(6.170831840931327) q[0];
rx(6.170831840931327) q[1];
rx(6.170831840931327) q[2];
rx(6.170831840931327) q[3];
rx(6.170831840931327) q[4];
rx(6.170831840931327) q[5];
rx(6.170831840931327) q[6];
rx(6.170831840931327) q[7];
cx q[0],q[2];
rz(3.706103888803324) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(4.11176199827019) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(3.0514275362646504) q[0];
cx q[4],q[0];
cx q[1],q[3];
rz(3.8853094416833964) q[3];
cx q[1],q[3];
cx q[1],q[5];
rz(4.043579359517348) q[5];
cx q[1],q[5];
cx q[7],q[1];
rz(3.907203703131753) q[1];
cx q[7],q[1];
cx q[2],q[5];
rz(4.04662788051126) q[5];
cx q[2],q[5];
cx q[6],q[2];
rz(4.335395543487458) q[2];
cx q[6],q[2];
cx q[6],q[3];
rz(3.7321106148764747) q[3];
cx q[6],q[3];
cx q[4],q[6];
rz(4.111442081043098) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(4.251346966458198) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(4.005043173011153) q[5];
cx q[7],q[5];
rx(0.575974544038209) q[0];
rx(0.575974544038209) q[1];
rx(0.575974544038209) q[2];
rx(0.575974544038209) q[3];
rx(0.575974544038209) q[4];
rx(0.575974544038209) q[5];
rx(0.575974544038209) q[6];
rx(0.575974544038209) q[7];
cx q[0],q[2];
rz(0.6491701571894095) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(0.7202262167573443) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(0.534495457440788) q[0];
cx q[4],q[0];
cx q[1],q[3];
rz(0.6805602370206405) q[3];
cx q[1],q[3];
cx q[1],q[5];
rz(0.7082831801763966) q[5];
cx q[1],q[5];
cx q[7],q[1];
rz(0.6843952890246913) q[1];
cx q[7],q[1];
cx q[2],q[5];
rz(0.7088171665168206) q[5];
cx q[2],q[5];
cx q[6],q[2];
rz(0.7593984116167808) q[2];
cx q[6],q[2];
cx q[6],q[3];
rz(0.653725558483986) q[3];
cx q[6],q[3];
cx q[4],q[6];
rz(0.7201701792789493) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(0.7446762587579605) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(0.7015330881654074) q[5];
cx q[7],q[5];
rx(1.0715104570115142) q[0];
rx(1.0715104570115142) q[1];
rx(1.0715104570115142) q[2];
rx(1.0715104570115142) q[3];
rx(1.0715104570115142) q[4];
rx(1.0715104570115142) q[5];
rx(1.0715104570115142) q[6];
rx(1.0715104570115142) q[7];
cx q[0],q[2];
rz(1.330003405516793) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(1.4755812638969705) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(1.0950607799151943) q[0];
cx q[4],q[0];
cx q[1],q[3];
rz(1.3943146074607236) q[3];
cx q[1],q[3];
cx q[1],q[5];
rz(1.4511126724976933) q[5];
cx q[1],q[5];
cx q[7],q[1];
rz(1.4021717650475798) q[1];
cx q[7],q[1];
cx q[2],q[5];
rz(1.4522066902115363) q[5];
cx q[2],q[5];
cx q[6],q[2];
rz(1.5558362663607033) q[2];
cx q[6],q[2];
cx q[6],q[3];
rz(1.3393363965179714) q[3];
cx q[6],q[3];
cx q[4],q[6];
rz(1.4754664557279928) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(1.5256738917938506) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(1.4372832546436192) q[5];
cx q[7],q[5];
rx(1.404443266496203) q[0];
rx(1.404443266496203) q[1];
rx(1.404443266496203) q[2];
rx(1.404443266496203) q[3];
rx(1.404443266496203) q[4];
rx(1.404443266496203) q[5];
rx(1.404443266496203) q[6];
rx(1.404443266496203) q[7];
