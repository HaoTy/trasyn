OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
cx q[0],q[4];
rz(5.483318043824546) q[4];
cx q[0],q[4];
cx q[2],q[4];
rz(4.619306379569584) q[4];
cx q[2],q[4];
h q[5];
h q[6];
h q[7];
cx q[1],q[7];
rz(4.804919300927596) q[7];
cx q[1],q[7];
h q[8];
cx q[5],q[8];
rz(4.857711019280911) q[8];
cx q[5],q[8];
cx q[6],q[8];
rz(5.184170267668894) q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(-4.585058963433594) q[8];
h q[8];
rz(0.5655560165405085) q[8];
h q[8];
rz(3*pi) q[8];
cx q[7],q[8];
h q[9];
h q[10];
cx q[9],q[10];
rz(4.1436239433955) q[10];
cx q[9],q[10];
h q[11];
cx q[2],q[11];
rz(6.138614422109076) q[11];
cx q[2],q[11];
h q[12];
cx q[3],q[12];
rz(5.4581810831791415) q[12];
cx q[3],q[12];
cx q[10],q[12];
rz(4.585030696085071) q[12];
cx q[10],q[12];
h q[13];
cx q[5],q[13];
rz(4.903074021996911) q[13];
cx q[5],q[13];
h q[14];
cx q[9],q[14];
rz(4.987836848824264) q[14];
cx q[9],q[14];
cx q[11],q[14];
rz(4.654619714281302) q[14];
cx q[11],q[14];
h q[15];
cx q[15],q[9];
rz(-4.238204686175762) q[9];
h q[9];
rz(0.5655560165405085) q[9];
h q[9];
rz(3*pi) q[9];
cx q[15],q[9];
cx q[15],q[11];
rz(-4.11729939154793) q[11];
h q[11];
rz(0.5655560165405089) q[11];
h q[11];
rz(3*pi) q[11];
cx q[15],q[11];
h q[16];
cx q[6],q[16];
rz(5.1793175749689135) q[16];
cx q[6],q[16];
cx q[16],q[14];
rz(-4.09371681239554) q[14];
h q[14];
rz(0.5655560165405085) q[14];
h q[14];
rz(3*pi) q[14];
cx q[16],q[14];
h q[17];
cx q[17],q[12];
rz(-3.436804195543743) q[12];
h q[12];
rz(0.5655560165405085) q[12];
h q[12];
rz(3*pi) q[12];
cx q[17],q[12];
h q[18];
cx q[0],q[18];
rz(5.205328782535628) q[18];
cx q[0],q[18];
cx q[1],q[18];
rz(5.013746296921112) q[18];
cx q[1],q[18];
cx q[18],q[15];
rz(-4.287777126464402) q[15];
h q[15];
rz(0.5655560165405085) q[15];
h q[15];
rz(3*pi) q[15];
cx q[18],q[15];
h q[18];
rz(5.717629290639078) q[18];
h q[18];
h q[19];
cx q[19],q[5];
rz(-3.953444521613846) q[5];
h q[5];
rz(0.5655560165405085) q[5];
h q[5];
rz(3*pi) q[5];
cx q[19],q[5];
cx q[13],q[19];
rz(4.47149371592684) q[19];
cx q[13],q[19];
h q[20];
cx q[20],q[6];
rz(-4.077247910213034) q[6];
h q[6];
rz(0.5655560165405085) q[6];
h q[6];
rz(3*pi) q[6];
cx q[20],q[6];
cx q[17],q[20];
rz(5.1334870948567595) q[20];
cx q[17],q[20];
h q[21];
cx q[21],q[1];
rz(-3.6857754021643916) q[1];
h q[1];
rz(0.5655560165405085) q[1];
h q[1];
rz(3*pi) q[1];
cx q[21],q[1];
cx q[3],q[21];
rz(6.2798864831889905) q[21];
cx q[3],q[21];
h q[22];
cx q[22],q[0];
rz(-4.555927835893991) q[0];
h q[0];
rz(0.5655560165405085) q[0];
h q[0];
rz(3*pi) q[0];
cx q[22],q[0];
cx q[22],q[17];
rz(-4.355405183770514) q[17];
h q[17];
rz(0.5655560165405085) q[17];
h q[17];
rz(3*pi) q[17];
cx q[22],q[17];
cx q[22],q[19];
rz(-4.21922687441811) q[19];
h q[19];
rz(0.5655560165405085) q[19];
h q[19];
rz(3*pi) q[19];
cx q[22],q[19];
h q[22];
rz(5.717629290639078) q[22];
h q[22];
h q[23];
cx q[23],q[3];
rz(-3.952687284914013) q[3];
h q[3];
rz(0.5655560165405085) q[3];
h q[3];
rz(3*pi) q[3];
cx q[23],q[3];
cx q[23],q[7];
rz(1.455468052283699) q[7];
h q[7];
rz(0.5655560165405085) q[7];
h q[7];
rz(3*pi) q[7];
cx q[23],q[7];
cx q[23],q[10];
rz(-4.696742195918175) q[10];
h q[10];
rz(0.5655560165405085) q[10];
h q[10];
rz(3*pi) q[10];
cx q[23],q[10];
h q[23];
rz(5.717629290639078) q[23];
h q[23];
h q[24];
cx q[24],q[4];
rz(-3.5874632471166925) q[4];
h q[4];
rz(0.5655560165405085) q[4];
h q[4];
rz(3*pi) q[4];
cx q[24],q[4];
cx q[24],q[13];
rz(-3.7541005928668487) q[13];
h q[13];
rz(0.5655560165405085) q[13];
h q[13];
rz(3*pi) q[13];
cx q[24],q[13];
cx q[24],q[21];
rz(-3.0244843725131134) q[21];
h q[21];
rz(0.5655560165405089) q[21];
h q[21];
rz(3*pi) q[21];
cx q[24],q[21];
h q[24];
rz(5.717629290639078) q[24];
h q[24];
h q[25];
cx q[25],q[2];
rz(-3.576464348284379) q[2];
h q[2];
rz(0.5655560165405085) q[2];
h q[2];
rz(3*pi) q[2];
cx q[25],q[2];
cx q[25],q[16];
rz(-4.012536811881656) q[16];
h q[16];
rz(0.5655560165405085) q[16];
h q[16];
rz(3*pi) q[16];
cx q[25],q[16];
cx q[25],q[20];
rz(-3.7251129371181864) q[20];
h q[20];
rz(0.5655560165405085) q[20];
h q[20];
rz(3*pi) q[20];
cx q[25],q[20];
h q[25];
rz(5.717629290639078) q[25];
h q[25];
