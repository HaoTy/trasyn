OPENQASM 2.0;
include "qelib1.inc";
gate mcphase(param0) q0,q1,q2,q3 { u(pi/2,0,pi) q3; cx q1,q3; p(-pi/4) q3; cx q0,q3; p(pi/4) q3; cx q1,q3; p(pi/4) q1; p(-pi/4) q3; cx q0,q3; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/2,-pi/4,-3*pi/4) q3; cx q2,q3; u(pi/2,0,-3*pi/4) q3; cx q1,q3; p(-pi/4) q3; cx q0,q3; p(pi/4) q3; cx q1,q3; p(pi/4) q1; p(-pi/4) q3; cx q0,q3; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/2,-pi/4,-3*pi/4) q3; cx q2,q3; u(0,-7*pi/8,-7*pi/8) q3; cx q0,q2; u(0,-pi/16,-pi/16) q2; cx q1,q2; u(0,-15*pi/16,-15*pi/16) q2; cx q0,q2; u(0,-pi/16,-pi/16) q2; cx q1,q2; u(0,-15*pi/16,-15*pi/16) q2; u(0,0,pi/8) q1; cx q0,q1; u(0,0,-pi/8) q1; cx q0,q1; p(pi/8) q0; }
gate gate_Q q0,q1,q2,q3 { mcphase(pi) q0,q1,q2,q3; h q2; h q1; h q0; x q0; x q1; x q2; h q2; ccx q0,q1,q2; h q2; x q0; x q1; x q2; h q0; h q1; h q2; }
gate gate_Q_140134536624656 q0,q1,q2,q3 { gate_Q q0,q1,q2,q3; }
gate gate_Q_140134562859408 q0,q1,q2,q3 { gate_Q q0,q1,q2,q3; }
qreg q[3];
qreg flag[1];
h q[0];
h q[1];
h q[2];
x flag[0];
gate_Q_140134536624656 q[0],q[1],q[2],flag[0];
gate_Q_140134562859408 q[0],q[1],q[2],flag[0];
