OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,2.073074917620143) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cx q[2],q[5];
u3(0,0,2.6856958975128746) q[5];
cx q[2],q[5];
u3(pi/2,0,pi) q[6];
cx q[3],q[6];
u3(0,0,2.344480121326011) q[6];
cx q[3],q[6];
u3(pi/2,0,pi) q[7];
cx q[2],q[7];
u3(0,0,2.0786693545539894) q[7];
cx q[2],q[7];
cx q[6],q[7];
u3(0,0,2.0898606789084213) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[8];
cx q[8],q[2];
u3(2.325356243131098,-pi/2,-2.2476100531661345) q[2];
cx q[8],q[2];
u3(pi/2,0,pi) q[9];
cx q[0],q[9];
u3(0,0,2.4549860607042855) q[9];
cx q[0],q[9];
cx q[1],q[9];
u3(0,0,2.346509853708015) q[9];
cx q[1],q[9];
u3(pi/2,0,pi) q[10];
cx q[10],q[6];
u3(2.325356243131098,-pi/2,-2.2430451000955163) q[6];
cx q[10],q[6];
cx q[10],q[7];
u3(2.325356243131098,-pi/2,-2.201729807939336) q[7];
cx q[10],q[7];
u3(pi/2,0,pi) q[11];
cx q[11],q[1];
u3(2.325356243131098,-pi/2,-2.1158705941999507) q[1];
cx q[11],q[1];
cx q[4],q[11];
u3(0,0,2.0471521550531446) q[11];
cx q[4],q[11];
cx q[11],q[10];
u3(2.325356243131098,-pi/2,-2.48307601033037) q[10];
cx q[11],q[10];
u3(2.325356243131098,-pi/2,pi/2) q[11];
u3(pi/2,0,pi) q[12];
cx q[3],q[12];
u3(0,0,2.273049652751677) q[12];
cx q[3],q[12];
cx q[4],q[12];
u3(0,0,2.1267990004734836) q[12];
cx q[4],q[12];
u3(pi/2,0,pi) q[13];
cx q[5],q[13];
u3(0,0,2.2348582289058854) q[13];
cx q[5],q[13];
cx q[8],q[13];
u3(0,0,2.5046849318911617) q[13];
cx q[8],q[13];
cx q[13],q[12];
u3(2.325356243131098,-pi/2,-2.362717513291879) q[12];
cx q[13],q[12];
u3(2.325356243131098,-pi/2,pi/2) q[13];
u3(pi/2,0,pi) q[14];
cx q[14],q[0];
u3(2.325356243131098,-pi/2,-2.4166315861192658) q[0];
cx q[14],q[0];
cx q[14],q[4];
u3(2.325356243131098,-pi/2,-2.948957579247266) q[4];
cx q[14],q[4];
cx q[14],q[8];
u3(2.325356243131098,-pi/2,-1.5939693252185343) q[8];
cx q[14],q[8];
u3(2.325356243131098,-pi/2,pi/2) q[14];
u3(pi/2,0,pi) q[15];
cx q[15],q[3];
u3(2.325356243131098,-pi/2,-2.269385908625379) q[3];
cx q[15],q[3];
cx q[15],q[5];
u3(2.325356243131098,-pi/2,-2.615011913700144) q[5];
cx q[15],q[5];
cx q[15],q[9];
u3(2.325356243131098,-pi/2,-2.341204648618824) q[9];
cx q[15],q[9];
u3(2.325356243131098,-pi/2,pi/2) q[15];
