OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(4.212066714765429) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(4.995840089520035) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(5.4078777391369295) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(5.49201111261538) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.348745151199968) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(4.538296987346143) q[2];
cx q[3],q[2];
rx(1.6997376997293516) q[0];
rx(1.6997376997293516) q[1];
rx(1.6997376997293516) q[2];
rx(1.6997376997293516) q[3];
cx q[0],q[1];
rz(5.073673824149398) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(6.017773413459711) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(6.514096187763561) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(6.615439619304809) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(6.442867623762365) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(5.466636734455549) q[2];
cx q[3],q[2];
rx(2.339910170174863) q[0];
rx(2.339910170174863) q[1];
rx(2.339910170174863) q[2];
rx(2.339910170174863) q[3];
