OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,3.8130491180918638) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cx q[0],q[5];
u3(0,0,3.698837605698952) q[5];
cx q[0],q[5];
cx q[3],q[5];
u3(0,0,3.2279419283023874) q[5];
cx q[3],q[5];
u3(pi/2,0,pi) q[6];
cx q[2],q[6];
u3(0,0,3.6228895777150845) q[6];
cx q[2],q[6];
cx q[4],q[6];
u3(0,0,3.7020016161121254) q[6];
cx q[4],q[6];
u3(pi/2,0,pi) q[7];
cx q[0],q[7];
u3(0,0,3.211846225869848) q[7];
cx q[0],q[7];
u3(pi/2,0,pi) q[8];
cx q[1],q[8];
u3(0,0,3.097440040602554) q[8];
cx q[1],q[8];
u3(pi/2,0,pi) q[9];
cx q[1],q[9];
u3(0,0,3.6777146115237698) q[9];
cx q[1],q[9];
cx q[9],q[5];
u3(1.2008166279264862,-pi/2,-1.0671775484169146) q[5];
cx q[9],q[5];
u3(pi/2,0,pi) q[10];
cx q[4],q[10];
u3(0,0,3.416208508137123) q[10];
cx q[4],q[10];
cx q[8],q[10];
u3(0,0,4.3385175720247275) q[10];
cx q[8],q[10];
cx q[10],q[9];
u3(1.2008166279264862,-pi/2,-1.1732548153885707) q[9];
cx q[10],q[9];
u3(1.2008166279264862,-pi/2,pi/2) q[10];
u3(pi/2,0,pi) q[11];
cx q[11],q[1];
u3(1.200816627926486,-pi/2,-0.5133132660856479) q[1];
cx q[11],q[1];
cx q[11],q[4];
u3(1.2008166279264862,-pi/2,-1.1797459995132058) q[4];
cx q[11],q[4];
cx q[11],q[8];
u3(1.2008166279264862,-pi/2,-1.1211944093131638) q[8];
cx q[11],q[8];
u3(1.2008166279264862,-pi/2,pi/2) q[11];
u3(pi/2,0,pi) q[12];
cx q[7],q[12];
u3(0,0,3.610710540737455) q[12];
cx q[7],q[12];
u3(pi/2,0,pi) q[13];
cx q[13],q[6];
u3(1.2008166279264862,-pi/2,-1.3444627542909247) q[6];
cx q[13],q[6];
cx q[13],q[7];
u3(1.2008166279264862,-pi/2,-0.7371577113576757) q[7];
cx q[13],q[7];
cx q[12],q[13];
u3(1.2008166279264862,-pi/2,-0.6148060976781347) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[14];
cx q[14],q[2];
u3(1.2008166279264862,-pi/2,-1.5388614624528998) q[2];
cx q[14],q[2];
cx q[14],q[3];
u3(1.2008166279264862,-pi/2,-0.700306880231174) q[3];
cx q[14],q[3];
u3(pi/2,0,pi) q[15];
cx q[15],q[0];
u3(1.2008166279264862,-pi/2,-0.6229762136087951) q[0];
cx q[15],q[0];
cx q[15],q[12];
u3(1.2008166279264862,-pi/2,-1.278645907295483) q[12];
cx q[15],q[12];
cx q[15],q[14];
u3(1.2008166279264862,-pi/2,-1.0117050835629042) q[14];
cx q[15],q[14];
u3(1.2008166279264862,-pi/2,pi/2) q[15];
