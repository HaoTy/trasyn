OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[4];
cx q[1],q[4];
u3(0,0,0.6783645549373608) q[4];
cx q[1],q[4];
cx q[2],q[4];
u3(0,0,0.7086984688691819) q[4];
cx q[2],q[4];
u3(pi/2,0,pi) q[5];
cx q[5],q[4];
u3(0.09134142427982195,-pi/2,2.2983880944747206) q[4];
cx q[5],q[4];
u3(pi/2,0,pi) q[6];
cx q[0],q[6];
u3(0,0,0.9962000479240763) q[6];
cx q[0],q[6];
cx q[3],q[6];
u3(0,0,0.767985936608367) q[6];
cx q[3],q[6];
u3(pi/2,0,pi) q[7];
cx q[0],q[7];
u3(0,0,0.8997065449680216) q[7];
cx q[0],q[7];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cx q[1],q[9];
u3(0,0,0.7718237735605424) q[9];
cx q[1],q[9];
u3(pi/2,0,pi) q[10];
cx q[10],q[1];
u3(0.09134142427982195,-pi/2,2.3038743977223133) q[1];
cx q[10],q[1];
cx q[3],q[10];
u3(0,0,0.7835193925769932) q[10];
cx q[3],q[10];
cx q[8],q[10];
u3(0.09134142427982193,-pi/2,2.384763798332928) q[10];
cx q[8],q[10];
u3(pi/2,0,pi) q[11];
cx q[11],q[3];
u3(0.09134142427982195,-pi/2,2.377323309829391) q[3];
cx q[11],q[3];
cx q[11],q[6];
u3(0.09134142427982195,-pi/2,2.376313224961219) q[6];
cx q[11],q[6];
u3(pi/2,0,pi) q[12];
cx q[5],q[12];
u3(0,0,0.6539349776520067) q[12];
cx q[5],q[12];
u3(pi/2,0,pi) q[13];
cx q[2],q[13];
u3(0,0,0.7630526980917727) q[13];
cx q[2],q[13];
cx q[7],q[13];
u3(0,0,0.7376557228584977) q[13];
cx q[7],q[13];
cx q[12],q[13];
u3(0.09134142427982195,-pi/2,2.4435314752073234) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[14];
cx q[8],q[14];
u3(0,0,0.7863167524366772) q[14];
cx q[8],q[14];
cx q[9],q[14];
u3(0,0,0.8047349402342494) q[14];
cx q[9],q[14];
cx q[14],q[12];
u3(0.09134142427982195,-pi/2,2.434603743259909) q[12];
cx q[14],q[12];
u3(0.09134142427982193,-pi/2,pi/2) q[14];
u3(pi/2,0,pi) q[15];
cx q[15],q[0];
u3(0.09134142427982195,-pi/2,2.458638501674354) q[0];
cx q[15],q[0];
cx q[15],q[5];
u3(0.09134142427982195,-pi/2,2.2891201696804027) q[5];
cx q[15],q[5];
cx q[15],q[8];
u3(0.09134142427982195,-pi/2,2.447842066531253) q[8];
cx q[15],q[8];
u3(0.09134142427982193,-pi/2,pi/2) q[15];
u3(pi/2,0,pi) q[16];
cx q[16],q[7];
u3(0.09134142427982195,-pi/2,2.2915203660086743) q[7];
cx q[16],q[7];
cx q[16],q[11];
u3(0.09134142427982196,-pi/2,2.3511150990801415) q[11];
cx q[16],q[11];
u3(pi/2,0,pi) q[17];
cx q[17],q[2];
u3(0.09134142427982195,-pi/2,2.229756960245414) q[2];
cx q[17],q[2];
cx q[17],q[9];
u3(0.09134142427982195,-pi/2,2.4815768345878526) q[9];
cx q[17],q[9];
cx q[17],q[16];
u3(0.09134142427982195,-pi/2,2.3092486572002873) q[16];
cx q[17],q[16];
u3(0.09134142427982193,-pi/2,pi/2) q[17];
