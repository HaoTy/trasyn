OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
h q[1];
cx q[0],q[1];
rz(2.018555494730038) q[1];
cx q[0],q[1];
h q[2];
h q[3];
cx q[2],q[3];
rz(2.0403289340254114) q[3];
cx q[2],q[3];
h q[4];
cx q[0],q[4];
rz(1.7445648690840065) q[4];
cx q[0],q[4];
h q[5];
cx q[1],q[5];
rz(1.7343570228221736) q[5];
cx q[1],q[5];
cx q[4],q[5];
rz(1.5484495797246434) q[5];
cx q[4],q[5];
h q[6];
cx q[2],q[6];
rz(1.770187188682258) q[6];
cx q[2],q[6];
cx q[3],q[6];
rz(2.0240150034354767) q[6];
cx q[3],q[6];
h q[7];
cx q[7],q[0];
rz(-1.5410516229328555) q[0];
h q[0];
rz(0.2661004819565833) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[7],q[2];
rz(-1.1220115588734654) q[2];
h q[2];
rz(0.2661004819565833) q[2];
h q[2];
rz(3*pi) q[2];
cx q[7],q[2];
cx q[7],q[4];
rz(-1.4486581025773084) q[4];
h q[4];
rz(0.2661004819565833) q[4];
h q[4];
rz(3*pi) q[4];
cx q[7],q[4];
h q[7];
rz(6.017084825223003) q[7];
h q[7];
h q[8];
cx q[8],q[1];
rz(-1.4204674508107304) q[1];
h q[1];
rz(0.2661004819565833) q[1];
h q[1];
rz(3*pi) q[1];
cx q[8],q[1];
cx q[0],q[1];
rz(7.350490555874924) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(7.205618822792133) q[4];
cx q[0],q[4];
cx q[7],q[0];
rz(0.846281336942317) q[0];
h q[0];
rz(1.6603043475155808) q[0];
h q[0];
cx q[7],q[0];
cx q[8],q[5];
rz(-1.1872520347903057) q[5];
h q[5];
rz(0.2661004819565833) q[5];
h q[5];
rz(3*pi) q[5];
cx q[8],q[5];
cx q[1],q[5];
rz(7.200221454267179) q[5];
cx q[1],q[5];
cx q[4],q[5];
rz(0.8187381363033721) q[5];
cx q[4],q[5];
h q[9];
cx q[9],q[3];
rz(-1.0539222946640234) q[3];
h q[3];
rz(0.26610048195658287) q[3];
h q[3];
rz(3*pi) q[3];
cx q[9],q[3];
cx q[2],q[3];
rz(7.362003197515332) q[3];
cx q[2],q[3];
cx q[9],q[6];
rz(-1.3650327775354727) q[6];
h q[6];
rz(0.26610048195658287) q[6];
h q[6];
rz(3*pi) q[6];
cx q[9],q[6];
cx q[2],q[6];
rz(7.219166548511026) q[6];
cx q[2],q[6];
cx q[3],q[6];
rz(1.070191947778829) q[6];
cx q[3],q[6];
cx q[7],q[2];
rz(1.06784753165525) q[2];
h q[2];
rz(1.6603043475155808) q[2];
h q[2];
cx q[7],q[2];
cx q[7],q[4];
rz(0.8951341376101052) q[4];
h q[4];
rz(1.6603043475155808) q[4];
h q[4];
cx q[7],q[4];
h q[7];
rz(1.6603043475155808) q[7];
h q[7];
cx q[9],q[8];
rz(-1.349559088234321) q[8];
h q[8];
rz(0.2661004819565833) q[8];
h q[8];
rz(3*pi) q[8];
cx q[9],q[8];
h q[9];
rz(6.017084825223003) q[9];
h q[9];
cx q[8],q[1];
rz(0.9100398613680909) q[1];
h q[1];
rz(1.6603043475155808) q[1];
h q[1];
cx q[8],q[1];
cx q[8],q[5];
rz(1.0333518229391823) q[5];
h q[5];
rz(1.6603043475155808) q[5];
h q[5];
cx q[8],q[5];
cx q[9],q[3];
rz(1.1038495287567773) q[3];
h q[3];
rz(1.6603043475155808) q[3];
h q[3];
cx q[9],q[3];
cx q[9],q[6];
rz(0.9393507809345145) q[6];
h q[6];
rz(1.6603043475155808) q[6];
h q[6];
cx q[9],q[6];
cx q[9],q[8];
rz(0.9475324483946945) q[8];
h q[8];
rz(1.6603043475155808) q[8];
h q[8];
cx q[9],q[8];
h q[9];
rz(1.6603043475155808) q[9];
h q[9];
