OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
cx q[0],q[1];
rz(5.554864891627152) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(5.383071442569184) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(4.738112260433671) q[0];
cx q[4],q[0];
cx q[1],q[2];
rz(5.689172028665363) q[2];
cx q[1],q[2];
cx q[5],q[1];
rz(4.574640808925641) q[1];
cx q[5],q[1];
cx q[2],q[4];
rz(4.370745629719437) q[4];
cx q[2],q[4];
cx q[7],q[2];
rz(4.601229757608023) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(4.94655056020646) q[4];
cx q[3],q[4];
cx q[6],q[3];
rz(5.203198701638453) q[3];
cx q[6],q[3];
cx q[5],q[6];
rz(5.008822525096217) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(5.796079686524396) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(5.941167637689) q[6];
cx q[7],q[6];
rx(3.830007695342652) q[0];
rx(3.830007695342652) q[1];
rx(3.830007695342652) q[2];
rx(3.830007695342652) q[3];
rx(3.830007695342652) q[4];
rx(3.830007695342652) q[5];
rx(3.830007695342652) q[6];
rx(3.830007695342652) q[7];
cx q[0],q[1];
rz(1.5208727059301377) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(1.4738371843058973) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(1.2972530844788361) q[0];
cx q[4],q[0];
cx q[1],q[2];
rz(1.5576448080276921) q[2];
cx q[1],q[2];
cx q[5],q[1];
rz(1.252496051923087) q[1];
cx q[5],q[1];
cx q[2],q[4];
rz(1.1966713614985076) q[4];
cx q[2],q[4];
cx q[7],q[2];
rz(1.2597758700859671) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(1.354321640992739) q[4];
cx q[3],q[4];
cx q[6],q[3];
rz(1.424589624273477) q[3];
cx q[6],q[3];
cx q[5],q[6];
rz(1.3713711522937653) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(1.5869151759060895) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(1.6266389692278989) q[6];
cx q[7],q[6];
rx(3.99859223271304) q[0];
rx(3.99859223271304) q[1];
rx(3.99859223271304) q[2];
rx(3.99859223271304) q[3];
rx(3.99859223271304) q[4];
rx(3.99859223271304) q[5];
rx(3.99859223271304) q[6];
rx(3.99859223271304) q[7];
cx q[0],q[1];
rz(3.571862285937354) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(3.461396495385694) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(3.04667796962799) q[0];
cx q[4],q[0];
cx q[1],q[2];
rz(3.658223809912869) q[2];
cx q[1],q[2];
cx q[5],q[1];
rz(2.9415633495858216) q[1];
cx q[5],q[1];
cx q[2],q[4];
rz(2.810455660182102) q[4];
cx q[2],q[4];
cx q[7],q[2];
rz(2.958660446432313) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(3.180706954387914) q[4];
cx q[3],q[4];
cx q[6],q[3];
rz(3.345735597752149) q[3];
cx q[6],q[3];
cx q[5],q[6];
rz(3.2207487712818224) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(3.7269670536522717) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(3.8202608044489663) q[6];
cx q[7],q[6];
rx(3.8623960738180796) q[0];
rx(3.8623960738180796) q[1];
rx(3.8623960738180796) q[2];
rx(3.8623960738180796) q[3];
rx(3.8623960738180796) q[4];
rx(3.8623960738180796) q[5];
rx(3.8623960738180796) q[6];
rx(3.8623960738180796) q[7];
