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
cx q[0],q[3];
rz(6.345336557018138) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(4.901757928026651) q[4];
cx q[0],q[4];
cx q[7],q[0];
rz(5.266189342254936) q[0];
cx q[7],q[0];
cx q[1],q[2];
rz(4.578565506808497) q[2];
cx q[1],q[2];
cx q[1],q[5];
rz(5.262368177457263) q[5];
cx q[1],q[5];
cx q[7],q[1];
rz(4.605211109661254) q[1];
cx q[7],q[1];
cx q[2],q[4];
rz(4.879790430024352) q[4];
cx q[2],q[4];
cx q[6],q[2];
rz(6.165098649887996) q[2];
cx q[6],q[2];
cx q[3],q[7];
rz(5.5840103151497695) q[7];
cx q[3],q[7];
cx q[9],q[3];
rz(4.862826300125089) q[3];
cx q[9],q[3];
cx q[8],q[4];
rz(5.337779292690254) q[4];
cx q[8],q[4];
cx q[5],q[6];
rz(4.575423787378473) q[6];
cx q[5],q[6];
cx q[8],q[5];
rz(5.607264790760097) q[5];
cx q[8],q[5];
cx q[9],q[6];
rz(5.566876601834802) q[6];
cx q[9],q[6];
cx q[9],q[8];
rz(5.43614407836933) q[8];
cx q[9],q[8];
rx(4.069930262953042) q[0];
rx(4.069930262953042) q[1];
rx(4.069930262953042) q[2];
rx(4.069930262953042) q[3];
rx(4.069930262953042) q[4];
rx(4.069930262953042) q[5];
rx(4.069930262953042) q[6];
rx(4.069930262953042) q[7];
rx(4.069930262953042) q[8];
rx(4.069930262953042) q[9];
cx q[0],q[3];
rz(6.7369105510074565) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(5.20424793973821) q[4];
cx q[0],q[4];
cx q[7],q[0];
rz(5.591168604634643) q[0];
cx q[7],q[0];
cx q[1],q[2];
rz(4.861111147395486) q[2];
cx q[1],q[2];
cx q[1],q[5];
rz(5.5871116337850655) q[5];
cx q[1],q[5];
cx q[7],q[1];
rz(4.889401064153038) q[1];
cx q[7],q[1];
cx q[2],q[4];
rz(5.180924816095976) q[4];
cx q[2],q[4];
cx q[6],q[2];
rz(6.545550069601067) q[2];
cx q[6],q[2];
cx q[3],q[7];
rz(5.928602473805618) q[7];
cx q[3],q[7];
cx q[9],q[3];
rz(5.162913820984834) q[3];
cx q[9],q[3];
cx q[8],q[4];
rz(5.667176407861465) q[4];
cx q[8],q[4];
cx q[5],q[6];
rz(4.857775550837881) q[6];
cx q[5],q[6];
cx q[8],q[5];
rz(5.953291995108327) q[5];
cx q[8],q[5];
cx q[9],q[6];
rz(5.91041143019866) q[6];
cx q[9],q[6];
cx q[9],q[8];
rz(5.7716113352343905) q[8];
cx q[9],q[8];
rx(3.0845659080174945) q[0];
rx(3.0845659080174945) q[1];
rx(3.0845659080174945) q[2];
rx(3.0845659080174945) q[3];
rx(3.0845659080174945) q[4];
rx(3.0845659080174945) q[5];
rx(3.0845659080174945) q[6];
rx(3.0845659080174945) q[7];
rx(3.0845659080174945) q[8];
rx(3.0845659080174945) q[9];
cx q[0],q[3];
rz(3.011474771124031) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(2.326357349490433) q[4];
cx q[0],q[4];
cx q[7],q[0];
rz(2.4993152375223433) q[0];
cx q[7],q[0];
cx q[1],q[2];
rz(2.1729713448283983) q[2];
cx q[1],q[2];
cx q[1],q[5];
rz(2.4975017259329895) q[5];
cx q[1],q[5];
cx q[7],q[1];
rz(2.18561725573185) q[1];
cx q[7],q[1];
cx q[2],q[4];
rz(2.3159316509598367) q[4];
cx q[2],q[4];
cx q[6],q[2];
rz(2.9259344841360564) q[2];
cx q[6],q[2];
cx q[3],q[7];
rz(2.6501519713987034) q[7];
cx q[3],q[7];
cx q[9],q[3];
rz(2.307880533616156) q[3];
cx q[9],q[3];
cx q[8],q[4];
rz(2.5332915802529) q[4];
cx q[8],q[4];
cx q[5],q[6];
rz(2.171480295659225) q[6];
cx q[5],q[6];
cx q[8],q[5];
rz(2.6611884650483217) q[5];
cx q[8],q[5];
cx q[9],q[6];
rz(2.6420203703527942) q[6];
cx q[9],q[6];
cx q[9],q[8];
rz(2.579975167132453) q[8];
cx q[9],q[8];
rx(3.4267830680673796) q[0];
rx(3.4267830680673796) q[1];
rx(3.4267830680673796) q[2];
rx(3.4267830680673796) q[3];
rx(3.4267830680673796) q[4];
rx(3.4267830680673796) q[5];
rx(3.4267830680673796) q[6];
rx(3.4267830680673796) q[7];
rx(3.4267830680673796) q[8];
rx(3.4267830680673796) q[9];
