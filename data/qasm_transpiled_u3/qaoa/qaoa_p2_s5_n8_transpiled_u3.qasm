OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,4.676345712779281) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,4.227018814702404) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[3];
cx q[0],q[3];
u3(0,0,5.426326562538032) q[3];
cx q[0],q[3];
u3(pi/2,0,pi) q[4];
cx q[4],q[0];
u3(2.3743130867482227,pi/2,2.35220838956674) q[0];
cx q[4],q[0];
cx q[2],q[4];
u3(0,0,4.492838753815453) q[4];
cx q[2],q[4];
cx q[3],q[4];
u3(2.374313086748223,pi/2,2.693819072884107) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[5];
cx q[5],q[1];
u3(2.374313086748223,pi/2,2.4369349925020662) q[1];
cx q[5],q[1];
cx q[0],q[1];
u3(0,0.3880999934585585,0.3880999934585585) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[6];
cx q[6],q[3];
u3(2.374313086748223,pi/2,2.6243660947044667) q[3];
cx q[6],q[3];
cx q[0],q[3];
u3(0,0.4503425180199909,0.4503425180199909) q[3];
cx q[0],q[3];
cx q[4],q[0];
u3(1.635598175551109,pi/2,-0.9196391206040953) q[0];
cx q[4],q[0];
cx q[5],q[6];
u3(0,0,4.893633101494334) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[7];
cx q[7],q[2];
u3(2.374313086748223,pi/2,-3.0697587789454897) q[2];
cx q[7],q[2];
cx q[1],q[2];
u3(0,0.3508093873068696,0.3508093873068696) q[2];
cx q[1],q[2];
cx q[2],q[4];
u3(0,0,0.7457407121125025) q[4];
cx q[2],q[4];
cx q[3],q[4];
u3(1.6355981755511089,pi/2,-0.8629371092029539) q[4];
cx q[3],q[4];
cx q[7],q[5];
u3(2.3743130867482227,pi/2,2.645026315625752) q[5];
cx q[7],q[5];
cx q[5],q[1];
u3(1.6355981755511089,pi/2,-0.9055758344124403) q[1];
cx q[5],q[1];
cx q[7],q[6];
u3(2.374313086748223,pi/2,2.7298035678562407) q[6];
cx q[7],q[6];
u3(3.9088722204313635,-pi/2,pi/2) q[7];
cx q[6],q[3];
u3(1.6355981755511089,pi/2,-0.874465213424819) q[3];
cx q[6],q[3];
cx q[5],q[6];
u3(0,0,0.8122662828321037) q[6];
cx q[5],q[6];
cx q[7],q[2];
u3(1.6355981755511089,pi/2,-0.776690427538139) q[2];
cx q[7],q[2];
cx q[7],q[5];
u3(1.6355981755511089,pi/2,-0.8710359410412609) q[5];
cx q[7],q[5];
cx q[7],q[6];
u3(1.6355981755511086,pi/2,-0.8569642478616166) q[6];
cx q[7],q[6];
u3(4.647587131628478,-pi/2,pi/2) q[7];
