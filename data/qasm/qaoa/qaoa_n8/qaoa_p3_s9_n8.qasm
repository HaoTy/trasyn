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
cx q[0],q[1];
rz(2.5415609796735414) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(2.493224419339645) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(3.0016604258022794) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(2.4016898520715113) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(2.858652900083701) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(2.6438763357547703) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(2.7523553610232105) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(2.9913748125933854) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(2.67475521778458) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(2.8956802826735353) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(2.5374074696681164) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(2.6732700578906337) q[5];
cx q[6],q[5];
rx(4.099709301812123) q[0];
rx(4.099709301812123) q[1];
rx(4.099709301812123) q[2];
rx(4.099709301812123) q[3];
rx(4.099709301812123) q[4];
rx(4.099709301812123) q[5];
rx(4.099709301812123) q[6];
rx(4.099709301812123) q[7];
cx q[0],q[1];
rz(4.865442840129336) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(4.772909639756057) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(5.7462352247378226) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(4.5976802399891925) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.472468453978915) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(5.061310466624231) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(5.268977526756833) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(5.726544938514596) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(5.120423522216717) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(5.543351905114358) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(4.857491559133552) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(5.117580403113835) q[5];
cx q[6],q[5];
rx(1.9588048197756918) q[0];
rx(1.9588048197756918) q[1];
rx(1.9588048197756918) q[2];
rx(1.9588048197756918) q[3];
rx(1.9588048197756918) q[4];
rx(1.9588048197756918) q[5];
rx(1.9588048197756918) q[6];
rx(1.9588048197756918) q[7];
cx q[0],q[1];
rz(1.5570075729778745) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.527395696224382) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(1.8388730594543736) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(1.4713199161956676) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(1.7512639867949187) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(1.6196878649419348) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(1.686144135400539) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(1.83257190131296) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(1.6386048429527005) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(1.7739476507168153) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(1.5544630554216925) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(1.637695006352103) q[5];
cx q[6],q[5];
rx(0.31084443540279794) q[0];
rx(0.31084443540279794) q[1];
rx(0.31084443540279794) q[2];
rx(0.31084443540279794) q[3];
rx(0.31084443540279794) q[4];
rx(0.31084443540279794) q[5];
rx(0.31084443540279794) q[6];
rx(0.31084443540279794) q[7];
