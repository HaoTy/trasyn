OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,4.438468665381447) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[3];
cx q[0],q[3];
u3(0,0,4.867334974917799) q[3];
cx q[0],q[3];
cx q[1],q[3];
u3(0,0,5.030861018679841) q[3];
cx q[1],q[3];
u3(pi/2,0,pi) q[4];
cx q[4],q[1];
u3(2.7630381175553347,-pi/2,-0.2824738893799177) q[1];
cx q[4],q[1];
cx q[2],q[4];
u3(0,0,4.9418849781116405) q[4];
cx q[2],q[4];
u3(pi/2,0,pi) q[5];
cx q[0],q[5];
u3(0,0,4.729958466291288) q[5];
cx q[0],q[5];
cx q[5],q[2];
u3(2.7630381175553347,-pi/2,-0.5506145600255516) q[2];
cx q[5],q[2];
u3(pi/2,0,pi) q[6];
cx q[6],q[0];
u3(2.7630381175553347,-pi/2,-1.0322014705531757) q[0];
cx q[6],q[0];
cx q[6],q[3];
u3(2.7630381175553347,-pi/2,0.400878329410304) q[3];
cx q[6],q[3];
u3(pi/2,0,pi) q[7];
cx q[7],q[4];
u3(2.7630381175553347,-pi/2,-0.10046347545440781) q[4];
cx q[7],q[4];
cx q[7],q[5];
u3(2.7630381175553347,-pi/2,-0.23296298771830148) q[5];
cx q[7],q[5];
cx q[7],q[6];
u3(2.7630381175553347,-pi/2,-0.4191907350663633) q[6];
cx q[7],q[6];
u3(2.7630381175553347,-pi/2,pi/2) q[7];
