OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.824852747925365) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[2];
cx q[0],q[2];
u3(0,0,0.7767612114350563) q[2];
cx q[0],q[2];
cx q[1],q[2];
u3(0,0,0.7964751994641986) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[3];
cx q[3],q[0];
u3(3.107321936872348,pi/2,-0.8425596924707852) q[0];
cx q[3],q[0];
cx q[3],q[1];
u3(3.107321936872348,pi/2,-0.6433383777141017) q[1];
cx q[3],q[1];
cx q[3],q[2];
u3(3.107321936872348,pi/2,-0.5443737186867925) q[2];
cx q[3],q[2];
u3(3.1758633703072388,-pi/2,pi/2) q[3];
