OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
h q[0];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.09641316507607714) q[2];
cx q[1],q[2];
h q[3];
cx q[1],q[3];
rz(0.09609675016317298) q[3];
cx q[1],q[3];
h q[4];
cx q[3],q[4];
rz(0.09185057586162551) q[4];
cx q[3],q[4];
h q[5];
h q[6];
cx q[0],q[6];
rz(0.11732934950478688) q[6];
cx q[0],q[6];
cx q[2],q[6];
rz(0.09126449002894446) q[6];
cx q[2],q[6];
h q[7];
cx q[7],q[1];
rz(-3.0684773288872407) q[1];
h q[1];
rz(0.8159888349899376) q[1];
h q[1];
rz(3*pi) q[1];
cx q[7],q[1];
cx q[5],q[7];
rz(0.10614843429708463) q[7];
cx q[5],q[7];
h q[8];
h q[9];
cx q[0],q[9];
rz(0.10003683751266398) q[9];
cx q[0],q[9];
cx q[4],q[9];
rz(0.11627433387818029) q[9];
cx q[4],q[9];
h q[10];
cx q[8],q[10];
rz(0.09896659321369039) q[10];
cx q[8],q[10];
h q[11];
cx q[11],q[0];
rz(-3.0617510025875063) q[0];
h q[0];
rz(0.815988834989938) q[0];
h q[0];
rz(3*pi) q[0];
cx q[11],q[0];
cx q[5],q[11];
rz(0.09115575382395756) q[11];
cx q[5],q[11];
cx q[11],q[9];
rz(-3.0640414582679805) q[9];
h q[9];
rz(0.815988834989938) q[9];
h q[9];
rz(3*pi) q[9];
cx q[11],q[9];
h q[11];
rz(5.467196472189649) q[11];
h q[11];
h q[12];
cx q[12],q[3];
rz(-3.0598646289992137) q[3];
h q[3];
rz(0.815988834989938) q[3];
h q[3];
rz(3*pi) q[3];
cx q[12],q[3];
cx q[8],q[12];
rz(0.09268311056011344) q[12];
cx q[8],q[12];
cx q[10],q[12];
rz(-3.0501563072410507) q[12];
h q[12];
rz(0.815988834989938) q[12];
h q[12];
rz(3*pi) q[12];
cx q[10],q[12];
h q[13];
cx q[13],q[2];
rz(-3.0463860878515696) q[2];
h q[2];
rz(0.8159888349899376) q[2];
h q[2];
rz(3*pi) q[2];
cx q[13],q[2];
cx q[13],q[6];
rz(-3.0376430347963854) q[6];
h q[6];
rz(0.8159888349899376) q[6];
h q[6];
rz(3*pi) q[6];
cx q[13],q[6];
cx q[13],q[8];
rz(-3.044429000754551) q[8];
h q[8];
rz(0.8159888349899376) q[8];
h q[8];
rz(3*pi) q[8];
cx q[13],q[8];
h q[13];
rz(5.467196472189649) q[13];
h q[13];
h q[14];
cx q[14],q[7];
rz(-3.0439269230576897) q[7];
h q[7];
rz(0.8159888349899376) q[7];
h q[7];
rz(3*pi) q[7];
cx q[14],q[7];
cx q[14],q[10];
rz(-3.0453570644286803) q[10];
h q[10];
rz(0.8159888349899376) q[10];
h q[10];
rz(3*pi) q[10];
cx q[14],q[10];
h q[15];
cx q[15],q[4];
rz(-3.042845036778472) q[4];
h q[4];
rz(0.8159888349899376) q[4];
h q[4];
rz(3*pi) q[4];
cx q[15],q[4];
cx q[15],q[5];
rz(-3.0405718343007533) q[5];
h q[5];
rz(0.815988834989938) q[5];
h q[5];
rz(3*pi) q[5];
cx q[15],q[5];
cx q[15],q[14];
rz(-3.035628205226062) q[14];
h q[14];
rz(0.8159888349899376) q[14];
h q[14];
rz(3*pi) q[14];
cx q[15],q[14];
h q[15];
rz(5.467196472189649) q[15];
h q[15];
