OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(4.7955849627524145) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(4.568455556030491) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(4.685661326999273) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(3.9472568982281806) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(3.748957580527381) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(5.112343044365938) q[2];
cx q[3],q[2];
rx(4.499741154298695) q[0];
rx(4.499741154298695) q[1];
rx(4.499741154298695) q[2];
rx(4.499741154298695) q[3];
