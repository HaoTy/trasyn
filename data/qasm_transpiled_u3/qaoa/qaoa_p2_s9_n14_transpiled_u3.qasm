OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cx q[1],q[3];
u3(0,0,1.0369437313422616) q[3];
cx q[1],q[3];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cx q[3],q[5];
u3(0,0,0.9388816141040099) q[5];
cx q[3],q[5];
u3(pi/2,0,pi) q[6];
cx q[0],q[6];
u3(0,0,1.0434484880076147) q[6];
cx q[0],q[6];
cx q[4],q[6];
u3(0,0,1.039732532574942) q[6];
cx q[4],q[6];
u3(pi/2,0,pi) q[7];
cx q[2],q[7];
u3(0,0,0.9232517954525479) q[7];
cx q[2],q[7];
u3(pi/2,0,pi) q[8];
cx q[2],q[8];
u3(0,0,1.1765548323717174) q[8];
cx q[2],q[8];
cx q[4],q[8];
u3(0,0,0.9783006195182019) q[8];
cx q[4],q[8];
cx q[8],q[6];
u3(0.7490560231501153,pi/2,-0.4236179115339844) q[6];
cx q[8],q[6];
u3(pi/2,0,pi) q[9];
cx q[0],q[9];
u3(0,0,0.9481731231853343) q[9];
cx q[0],q[9];
cx q[1],q[9];
u3(0,0,1.171226253621837) q[9];
cx q[1],q[9];
cx q[9],q[4];
u3(0.7490560231501154,pi/2,-0.7161666054788611) q[4];
cx q[9],q[4];
u3(pi/2,0,pi) q[10];
cx q[10],q[2];
u3(0.7490560231501153,pi/2,-0.5236394285487611) q[2];
cx q[10],q[2];
cx q[7],q[10];
u3(0,0,0.9512237791233012) q[10];
cx q[7],q[10];
u3(pi/2,0,pi) q[11];
cx q[11],q[7];
u3(0.7490560231501153,pi/2,-0.274228601771187) q[7];
cx q[11],q[7];
cx q[11],q[10];
u3(0.7490560231501154,pi/2,-0.7196161621305901) q[10];
cx q[11],q[10];
cx q[2],q[7];
u3(0,1.8943256384813694,1.8943256384813694) q[7];
cx q[2],q[7];
cx q[2],q[8];
u3(0.7490560231501153,0.11571532446189803,-pi/2) q[8];
cx q[2],q[8];
cx q[10],q[2];
u3(0.10280116632042392,-pi/2,-0.4152813665555555) q[2];
cx q[10],q[2];
cx q[7],q[10];
u3(0,0,3.9034369640043045) q[10];
cx q[7],q[10];
u3(pi/2,0,pi) q[12];
cx q[12],q[0];
u3(0.7490560231501153,pi/2,-0.47571530467190737) q[0];
cx q[12],q[0];
cx q[0],q[6];
cx q[5],q[12];
u3(0,0,1.0679874482621228) q[12];
cx q[5],q[12];
cx q[12],q[11];
u3(0.7490560231501153,pi/2,-0.7076614072123855) q[11];
cx q[12],q[11];
u3(5.534129284029471,-pi/2,pi/2) q[12];
cx q[11],q[7];
u3(0,2.1409449004088463,2.1409449004088463) q[6];
cx q[0],q[6];
cx q[0],q[9];
cx q[4],q[6];
u3(0,0,4.266641025386684) q[6];
cx q[4],q[6];
cx q[4],q[8];
u3(0,0,4.01454934574408) q[8];
cx q[4],q[8];
u3(0.10280116632042392,-pi/2,0.6081996079883547) q[7];
cx q[11],q[7];
cx q[11],q[10];
u3(0.10280116632042391,-pi/2,-1.2194904735368635) q[10];
cx q[11],q[10];
cx q[8],q[6];
u3(0.10280116632042391,-pi/2,-0.004833588356502716) q[6];
cx q[8],q[6];
u3(0.10280116632042391,-pi/2,pi/2) q[8];
u3(0.7490560231501153,-0.8214706722572327,-pi/2) q[9];
cx q[0],q[9];
cx q[12],q[0];
u3(0.10280116632042391,-pi/2,-0.21862018049811205) q[0];
cx q[12],q[0];
u3(pi/2,0,pi) q[13];
cx q[13],q[1];
u3(0.7490560231501153,pi/2,-0.5849699995670536) q[1];
cx q[13],q[1];
cx q[13],q[3];
u3(0.7490560231501153,pi/2,-0.37066679173698525) q[3];
cx q[13],q[3];
cx q[1],q[3];
cx q[13],q[5];
u3(0,2.127598457559828,2.127598457559828) q[3];
cx q[1],q[3];
cx q[1],q[9];
u3(0,0,4.806237976739166) q[9];
cx q[1],q[9];
u3(0.7490560231501153,pi/2,-0.3983530660283954) q[5];
cx q[13],q[5];
u3(5.534129284029471,-pi/2,pi/2) q[13];
cx q[13],q[1];
u3(0.10280116632042391,-pi/2,-0.6669571795350624) q[1];
cx q[13],q[1];
cx q[3],q[5];
u3(0,1.9263948598380045,1.9263948598380045) q[5];
cx q[3],q[5];
cx q[13],q[3];
u3(0.10280116632042392,-pi/2,0.21245635207320746) q[3];
cx q[13],q[3];
cx q[5],q[12];
u3(0,0,4.382587750782696) q[12];
cx q[5],q[12];
cx q[12],q[11];
u3(0.10280116632042392,-pi/2,-1.170433003289074) q[11];
cx q[12],q[11];
u3(0.10280116632042391,-pi/2,pi/2) q[12];
cx q[13],q[5];
u3(0.10280116632042391,-pi/2,0.09884310055813561) q[5];
cx q[13],q[5];
u3(0.10280116632042391,-pi/2,pi/2) q[13];
cx q[9],q[4];
u3(0.10280116632042391,-pi/2,-1.2053348907609633) q[4];
cx q[9],q[4];
u3(0.10280116632042391,-pi/2,pi/2) q[9];
