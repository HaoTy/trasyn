OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,2.8001987611832853) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,2.4374342926336183) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cx q[1],q[5];
u3(0,0,2.1357451942723538) q[5];
cx q[1],q[5];
u3(pi/2,0,pi) q[6];
cx q[2],q[6];
u3(0,0,2.6683396652527147) q[6];
cx q[2],q[6];
u3(pi/2,0,pi) q[7];
cx q[0],q[7];
u3(0,0,2.114022543025729) q[7];
cx q[0],q[7];
cx q[7],q[2];
u3(1.503377789375737,pi/2,0.300470637817555) q[2];
cx q[7],q[2];
u3(pi/2,0,pi) q[8];
cx q[3],q[8];
u3(0,0,2.511499929816017) q[8];
cx q[3],q[8];
cx q[4],q[8];
u3(0,0,2.4757475392902335) q[8];
cx q[4],q[8];
cx q[5],q[8];
u3(1.503377789375737,pi/2,0.6930448526642095) q[8];
cx q[5],q[8];
u3(pi/2,0,pi) q[9];
cx q[9],q[0];
u3(1.503377789375737,pi/2,0.33609740101003416) q[0];
cx q[9],q[0];
cx q[4],q[9];
u3(0,0,2.497161465872061) q[9];
cx q[4],q[9];
cx q[9],q[7];
u3(1.503377789375737,pi/2,0.7228921494963187) q[7];
cx q[9],q[7];
u3(4.7798075178038495,-pi/2,pi/2) q[9];
u3(pi/2,0,pi) q[10];
cx q[10],q[1];
u3(1.503377789375737,pi/2,0.1172619238613537) q[1];
cx q[10],q[1];
cx q[10],q[4];
u3(1.503377789375737,pi/2,0.7833722091517403) q[4];
cx q[10],q[4];
cx q[6],q[10];
u3(1.503377789375737,pi/2,0.8753299258831504) q[10];
cx q[6],q[10];
u3(pi/2,0,pi) q[11];
cx q[11],q[3];
u3(1.503377789375737,pi/2,0.395692897604059) q[3];
cx q[11],q[3];
cx q[11],q[5];
u3(1.503377789375737,pi/2,0.9150634240730371) q[5];
cx q[11],q[5];
cx q[11],q[6];
u3(1.503377789375737,pi/2,0.7658072315511117) q[6];
cx q[11],q[6];
u3(4.7798075178038495,-pi/2,pi/2) q[11];
