OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(1.0819121954045705) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.4012586935933946) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(1.2717872413396853) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(1.0725954558505697) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(1.1294301403303872) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(1.2173239671496363) q[2];
cx q[3],q[2];
rx(3.87280348180506) q[0];
rx(3.87280348180506) q[1];
rx(3.87280348180506) q[2];
rx(3.87280348180506) q[3];
cx q[0],q[1];
rz(4.791616769305436) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(6.20595153920637) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(5.632543101441955) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(4.750354413938872) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.002066177969633) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(5.3913339358287224) q[2];
cx q[3],q[2];
rx(4.275296338906791) q[0];
rx(4.275296338906791) q[1];
rx(4.275296338906791) q[2];
rx(4.275296338906791) q[3];
cx q[0],q[1];
rz(5.241125894676115) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(6.7881416397068) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(6.160940852147055) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(5.195992652683065) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.471317894239779) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(5.897103474327713) q[2];
cx q[3],q[2];
rx(1.6318963398929296) q[0];
rx(1.6318963398929296) q[1];
rx(1.6318963398929296) q[2];
rx(1.6318963398929296) q[3];
