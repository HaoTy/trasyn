OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.0819121954045705) q[1];
cx q[0],q[1];
h q[2];
cx q[0],q[2];
rz(1.4012586935933946) q[2];
cx q[0],q[2];
cx q[1],q[2];
rz(1.0725954558505697) q[2];
cx q[1],q[2];
h q[3];
cx q[3],q[0];
rz(-1.8698054122501078) q[0];
h q[0];
rz(2.410381825374527) q[0];
h q[0];
rz(3*pi) q[0];
cx q[3],q[0];
cx q[3],q[1];
rz(-2.0121625132594056) q[1];
h q[1];
rz(2.410381825374527) q[1];
h q[1];
rz(3*pi) q[1];
cx q[3],q[1];
cx q[0],q[1];
rz(11.074802076485021) q[1];
cx q[0],q[1];
cx q[3],q[2];
rz(-1.9242686864401568) q[2];
h q[2];
rz(2.410381825374527) q[2];
h q[2];
rz(3*pi) q[2];
cx q[3],q[2];
h q[3];
rz(3.87280348180506) q[3];
h q[3];
cx q[0],q[2];
rz(12.489136846385957) q[2];
cx q[0],q[2];
cx q[1],q[2];
rz(4.750354413938872) q[2];
cx q[1],q[2];
cx q[3],q[0];
rz(-3.7922348593274244) q[0];
h q[0];
rz(2.0078889682727965) q[0];
h q[0];
rz(3*pi) q[0];
cx q[3],q[0];
cx q[3],q[1];
rz(-4.422711782799746) q[1];
h q[1];
rz(2.0078889682727956) q[1];
h q[1];
rz(3*pi) q[1];
cx q[3],q[1];
cx q[0],q[1];
rz(11.524311201855703) q[1];
cx q[0],q[1];
cx q[3],q[2];
rz(-4.033444024940657) q[2];
h q[2];
rz(2.0078889682727965) q[2];
h q[2];
rz(3*pi) q[2];
cx q[3],q[2];
h q[3];
rz(4.275296338906791) q[3];
h q[3];
cx q[0],q[2];
rz(0.5049563325272128) q[2];
cx q[0],q[2];
cx q[1],q[2];
rz(5.195992652683065) q[2];
cx q[1],q[2];
cx q[3],q[0];
rz(-0.12224445503253101) q[0];
h q[0];
rz(1.6318963398929291) q[0];
h q[0];
cx q[3],q[0];
cx q[3],q[1];
rz(-0.8118674129398071) q[1];
h q[1];
rz(1.6318963398929291) q[1];
h q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(-0.3860818328518736) q[2];
h q[2];
rz(1.6318963398929291) q[2];
h q[2];
cx q[3],q[2];
h q[3];
rz(1.6318963398929291) q[3];
h q[3];
