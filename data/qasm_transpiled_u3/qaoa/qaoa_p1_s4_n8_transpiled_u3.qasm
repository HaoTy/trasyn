OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cx q[0],q[2];
u3(0,0,4.177672023814229) q[2];
cx q[0],q[2];
cx q[1],q[2];
u3(0,0,4.874659410981916) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[3];
cx q[1],q[3];
u3(0,0,5.734547686954613) q[3];
cx q[1],q[3];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,4.374308462250765) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[5];
cx q[0],q[5];
u3(0,0,5.067076862193546) q[5];
cx q[0],q[5];
cx q[5],q[3];
u3(0.5588326340528991,pi/2,-2.4390550308864043) q[3];
cx q[5],q[3];
u3(pi/2,0,pi) q[6];
cx q[6],q[0];
u3(0.558832634052899,pi/2,-2.2403423906077333) q[0];
cx q[6],q[0];
cx q[4],q[6];
u3(0,0,5.835164900253404) q[6];
cx q[4],q[6];
cx q[6],q[5];
u3(0.558832634052899,pi/2,-1.683203262920164) q[5];
cx q[6],q[5];
u3(5.724352673126687,-pi/2,pi/2) q[6];
u3(pi/2,0,pi) q[7];
cx q[7],q[1];
u3(0.5588326340528991,pi/2,-1.5706141613339395) q[1];
cx q[7],q[1];
cx q[7],q[2];
u3(0.5588326340528991,pi/2,-2.1732946369273054) q[2];
cx q[7],q[2];
cx q[7],q[4];
u3(0.5588326340528991,pi/2,-1.7798031248549013) q[4];
cx q[7],q[4];
u3(5.724352673126687,-pi/2,pi/2) q[7];
