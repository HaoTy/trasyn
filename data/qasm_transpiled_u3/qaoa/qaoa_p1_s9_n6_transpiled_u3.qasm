OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,1.59238785227761) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,1.3565625978671851) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[3];
cx q[3],q[1];
u3(0.6128596497603529,-pi/2,3.0403299680948006) q[1];
cx q[3],q[1];
cx q[2],q[3];
u3(0,0,1.3417857016681398) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,1.3755475728072541) q[4];
cx q[0],q[4];
cx q[4],q[2];
u3(0.6128596497603529,-pi/2,2.994571605985623) q[2];
cx q[4],q[2];
u3(pi/2,0,pi) q[5];
cx q[5],q[0];
u3(0.6128596497603529,-pi/2,2.981712510535595) q[0];
cx q[5],q[0];
cx q[5],q[3];
u3(0.6128596497603529,-pi/2,2.7639378730402164) q[3];
cx q[5],q[3];
cx q[5],q[4];
u3(0.6128596497603529,-pi/2,2.6303878018628764) q[4];
cx q[5],q[4];
u3(0.6128596497603528,-pi/2,pi/2) q[5];
