OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
cx q[0],q[3];
rz(3.924593207371463) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(3.771203475135564) q[4];
cx q[0],q[4];
cx q[6],q[0];
rz(4.093113297102392) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(3.622104260813935) q[2];
cx q[1],q[2];
cx q[1],q[6];
rz(3.291738539711026) q[6];
cx q[1],q[6];
cx q[11],q[1];
rz(4.119153410066434) q[1];
cx q[11],q[1];
cx q[2],q[4];
rz(3.2905599605038027) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(4.372692867898637) q[2];
cx q[5],q[2];
cx q[3],q[5];
rz(3.1737335488079936) q[5];
cx q[3],q[5];
cx q[8],q[3];
rz(3.113366721392438) q[3];
cx q[8],q[3];
cx q[9],q[4];
rz(3.72817932079904) q[4];
cx q[9],q[4];
cx q[7],q[5];
rz(3.5499783410033228) q[5];
cx q[7],q[5];
cx q[9],q[6];
rz(3.8764531432440834) q[6];
cx q[9],q[6];
cx q[7],q[9];
rz(4.074851653555656) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(3.851132466798941) q[7];
cx q[10],q[7];
cx q[8],q[10];
rz(3.4984975431781336) q[10];
cx q[8],q[10];
cx q[11],q[8];
rz(3.4864938624886888) q[8];
cx q[11],q[8];
cx q[11],q[10];
rz(3.9920852357087635) q[10];
cx q[11],q[10];
rx(0.1715931981720197) q[0];
rx(0.1715931981720197) q[1];
rx(0.1715931981720197) q[2];
rx(0.1715931981720197) q[3];
rx(0.1715931981720197) q[4];
rx(0.1715931981720197) q[5];
rx(0.1715931981720197) q[6];
rx(0.1715931981720197) q[7];
rx(0.1715931981720197) q[8];
rx(0.1715931981720197) q[9];
rx(0.1715931981720197) q[10];
rx(0.1715931981720197) q[11];
