OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
s q[0];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
y q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
y q[1];
cx q[0],q[1];
s q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
s q[0];
x q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
y q[1];
h q[1];
t q[1];
h q[1];
t q[1];
y q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
y q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
y q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
cx q[0],q[1];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
z q[1];
s q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
y q[1];
h q[1];
t q[1];
h q[1];
t q[1];
y q[1];
h q[1];
cx q[0],q[1];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
x q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
h q[0];
t q[0];
y q[0];
h q[0];
x q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
x q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
h q[1];
t q[1];
y q[1];
