OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[1];
h q[2];
cx q[0],q[2];
rz(1.8458790782214876) q[2];
cx q[0],q[2];
h q[3];
cx q[0],q[3];
rz(2.333397787402257) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(1.8456130642326516) q[3];
cx q[2],q[3];
h q[4];
cx q[1],q[4];
rz(1.9218886401363673) q[4];
cx q[1],q[4];
cx q[4],q[2];
rz(-0.7912486064862998) q[2];
h q[2];
rz(1.1401774978052996) q[2];
h q[2];
rz(3*pi) q[2];
cx q[4],q[2];
h q[5];
cx q[1],q[5];
rz(1.9458599653214486) q[5];
cx q[1],q[5];
h q[6];
cx q[6],q[1];
rz(-0.7804830709128483) q[1];
h q[1];
rz(1.1401774978052996) q[1];
h q[1];
rz(3*pi) q[1];
cx q[6],q[1];
cx q[6],q[4];
rz(-1.3011161588006854) q[4];
h q[4];
rz(1.1401774978052996) q[4];
h q[4];
rz(3*pi) q[4];
cx q[6],q[4];
cx q[1],q[4];
rz(10.56466309720965) q[4];
cx q[1],q[4];
cx q[5],q[6];
rz(-1.1819620982474484) q[6];
h q[6];
rz(1.1401774978052996) q[6];
h q[6];
rz(3*pi) q[6];
cx q[5],q[6];
h q[7];
cx q[7],q[0];
rz(-0.8845315880143687) q[0];
h q[0];
rz(1.1401774978052996) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[0],q[2];
rz(10.395333176436829) q[2];
cx q[0],q[2];
cx q[7],q[3];
rz(-0.7798884192434188) q[3];
h q[3];
rz(1.1401774978052996) q[3];
h q[3];
rz(3*pi) q[3];
cx q[7],q[3];
cx q[0],q[3];
rz(11.48140053850495) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(4.111555257926258) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(-1.0472181307000135) q[2];
h q[2];
rz(0.9465761035159392) q[2];
h q[2];
cx q[4],q[2];
cx q[7],q[5];
rz(-1.100618105620978) q[5];
h q[5];
rz(1.1401774978052996) q[5];
h q[5];
rz(3*pi) q[5];
cx q[7],q[5];
h q[7];
rz(5.143007809374286) q[7];
h q[7];
cx q[1],q[5];
rz(10.618065096802212) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(-1.023235262758071) q[1];
h q[1];
rz(0.9465761035159392) q[1];
h q[1];
cx q[6],q[1];
cx q[6],q[4];
rz(-2.183073016098649) q[4];
h q[4];
rz(0.9465761035159392) q[4];
h q[4];
cx q[6],q[4];
cx q[1],q[4];
rz(7.493054802232823) q[4];
cx q[1],q[4];
cx q[5],q[6];
rz(-1.917628154624062) q[6];
h q[6];
rz(0.9465761035159392) q[6];
h q[6];
cx q[5],q[6];
cx q[7],q[0];
rz(-1.2550288254819302) q[0];
h q[0];
rz(0.9465761035159392) q[0];
h q[0];
cx q[7],q[0];
cx q[0],q[2];
rz(7.445205177456099) q[2];
cx q[0],q[2];
cx q[7],q[3];
rz(-1.0219105304805094) q[3];
h q[3];
rz(0.9465761035159392) q[3];
h q[3];
cx q[7],q[3];
cx q[0],q[3];
rz(7.752108502751621) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(1.1618524088515219) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(-1.6620014295796888) q[2];
h q[2];
rz(1.5904833058419179) q[2];
h q[2];
rz(3*pi) q[2];
cx q[4],q[2];
cx q[7],q[5];
rz(-1.7364144825572505) q[5];
h q[5];
rz(0.9465761035159392) q[5];
h q[5];
cx q[7],q[5];
h q[7];
rz(0.9465761035159392) q[7];
h q[7];
cx q[1],q[5];
rz(7.5081452577835535) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(-1.6552242975356046) q[1];
h q[1];
rz(1.5904833058419179) q[1];
h q[1];
rz(3*pi) q[1];
cx q[6],q[1];
cx q[6],q[4];
rz(-1.9829738237028285) q[4];
h q[4];
rz(1.5904833058419179) q[4];
h q[4];
rz(3*pi) q[4];
cx q[6],q[4];
cx q[1],q[4];
rz(6.793337921490497) q[4];
cx q[1],q[4];
cx q[5],q[6];
rz(-1.9079638256912586) q[6];
h q[6];
rz(1.5904833058419179) q[6];
h q[6];
rz(3*pi) q[6];
cx q[5],q[6];
cx q[7],q[0];
rz(-1.72072503686966) q[0];
h q[0];
rz(1.5904833058419179) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[0],q[2];
rz(6.773161686554871) q[2];
cx q[0],q[2];
cx q[7],q[3];
rz(-1.654849951748861) q[3];
h q[3];
rz(1.5904833058419179) q[3];
h q[3];
rz(3*pi) q[3];
cx q[7],q[3];
cx q[0],q[3];
rz(6.902570299946958) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(0.48990576772327993) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(-2.517709388389342) q[2];
h q[2];
rz(1.2276135214817812) q[2];
h q[2];
rz(3*pi) q[2];
cx q[4],q[2];
cx q[7],q[5];
rz(-1.8567560641023149) q[5];
h q[5];
rz(1.5904833058419179) q[5];
h q[5];
rz(3*pi) q[5];
cx q[7],q[5];
h q[7];
rz(4.692702001337668) q[7];
h q[7];
cx q[1],q[5];
rz(6.799700951041885) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(-2.514851748252157) q[1];
h q[1];
rz(1.2276135214817812) q[1];
h q[1];
rz(3*pi) q[1];
cx q[6],q[1];
cx q[6],q[4];
rz(-2.653050354215037) q[4];
h q[4];
rz(1.227613521481782) q[4];
h q[4];
rz(3*pi) q[4];
cx q[6],q[4];
cx q[1],q[4];
rz(9.328670123983443) q[4];
cx q[1],q[4];
cx q[5],q[6];
rz(-2.6214216977843083) q[6];
h q[6];
rz(1.2276135214817812) q[6];
h q[6];
rz(3*pi) q[6];
cx q[5],q[6];
cx q[7],q[0];
rz(-2.542470738190485) q[0];
h q[0];
rz(1.2276135214817812) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[0],q[2];
rz(9.208222996137753) q[2];
cx q[0],q[2];
cx q[7],q[3];
rz(-2.514693901903816) q[3];
h q[3];
rz(1.2276135214817812) q[3];
h q[3];
rz(3*pi) q[3];
cx q[7],q[3];
cx q[0],q[3];
rz(9.980760382389487) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(2.9246161548759435) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(-2.5587566591122757) q[2];
h q[2];
rz(1.711876490401231) q[2];
h q[2];
cx q[4],q[2];
cx q[7],q[5];
rz(-2.5998294735366345) q[5];
h q[5];
rz(1.2276135214817812) q[5];
h q[5];
rz(3*pi) q[5];
cx q[7],q[5];
h q[7];
rz(5.055571785697804) q[7];
h q[7];
cx q[1],q[5];
rz(9.366655835207922) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(-2.5416972549152765) q[1];
h q[1];
rz(1.711876490401231) q[1];
h q[1];
cx q[6],q[1];
cx q[6],q[4];
rz(-3.3667087208114523) q[4];
h q[4];
rz(1.711876490401231) q[4];
h q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(-3.1778934717397576) q[6];
h q[6];
rz(1.711876490401231) q[6];
h q[6];
cx q[5],q[6];
cx q[7],q[0];
rz(-2.70657578763877) q[0];
h q[0];
rz(1.711876490401231) q[0];
h q[0];
cx q[7],q[0];
cx q[7],q[3];
rz(-2.5407549512912713) q[3];
h q[3];
rz(1.711876490401231) q[3];
h q[3];
cx q[7],q[3];
cx q[7],q[5];
rz(-3.0489932382358313) q[5];
h q[5];
rz(1.711876490401231) q[5];
h q[5];
cx q[7],q[5];
h q[7];
rz(1.711876490401231) q[7];
h q[7];
