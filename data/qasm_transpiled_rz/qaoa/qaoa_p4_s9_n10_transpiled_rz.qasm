OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
h q[1];
h q[2];
cx q[1],q[2];
rz(5.2893076320927666) q[2];
cx q[1],q[2];
h q[3];
cx q[0],q[3];
rz(5.770960164050096) q[3];
cx q[0],q[3];
h q[4];
cx q[3],q[4];
rz(4.38815317016296) q[4];
cx q[3],q[4];
h q[5];
cx q[1],q[5];
rz(4.976691104239731) q[5];
cx q[1],q[5];
h q[6];
cx q[0],q[6];
rz(5.229781353547161) q[6];
cx q[0],q[6];
cx q[2],q[6];
rz(5.502124781258296) q[6];
cx q[2],q[6];
cx q[5],q[6];
rz(-3.461914977533272) q[6];
h q[6];
rz(2.7083713851400066) q[6];
h q[6];
rz(3*pi) q[6];
cx q[5],q[6];
h q[7];
cx q[7],q[0];
rz(-3.6275343301275824) q[0];
h q[0];
rz(2.7083713851400066) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[7],q[1];
rz(-4.026134198156253) q[1];
h q[1];
rz(2.7083713851400066) q[1];
h q[1];
rz(3*pi) q[1];
cx q[7],q[1];
cx q[4],q[7];
rz(-3.799210579950269) q[7];
h q[7];
rz(2.7083713851400066) q[7];
h q[7];
rz(3*pi) q[7];
cx q[4],q[7];
h q[8];
cx q[8],q[2];
rz(-4.682971521422277) q[2];
h q[2];
rz(2.7083713851400066) q[2];
h q[2];
rz(3*pi) q[2];
cx q[8],q[2];
cx q[1],q[2];
rz(10.31249193177311) q[2];
cx q[1],q[2];
cx q[8],q[4];
rz(-4.2539654114197205) q[4];
h q[4];
rz(2.7083713851400066) q[4];
h q[4];
rz(3*pi) q[4];
cx q[8],q[4];
h q[9];
cx q[9],q[3];
rz(1.0164273831239514) q[3];
h q[3];
rz(2.7083713851400066) q[3];
h q[3];
rz(3*pi) q[3];
cx q[9],q[3];
cx q[0],q[3];
rz(10.679406823618798) q[3];
cx q[0],q[3];
cx q[0],q[6];
rz(10.267145802342874) q[6];
cx q[0],q[6];
cx q[2],q[6];
cx q[3],q[4];
rz(9.626007443481182) q[4];
cx q[3],q[4];
rz(4.191427190952127) q[6];
cx q[2],q[6];
cx q[7],q[0];
rz(-1.8669414828275839) q[0];
h q[0];
rz(0.8480621575092626) q[0];
h q[0];
cx q[7],q[0];
cx q[9],q[5];
rz(-4.471349360128129) q[5];
h q[5];
rz(2.7083713851400066) q[5];
h q[5];
rz(3*pi) q[5];
cx q[9],q[5];
cx q[1],q[5];
rz(10.074345857796828) q[5];
cx q[1],q[5];
cx q[5],q[6];
rz(-1.7407754132800068) q[6];
h q[6];
rz(0.8480621575092626) q[6];
h q[6];
cx q[5],q[6];
cx q[7],q[1];
rz(-2.1705882359612385) q[1];
h q[1];
rz(0.8480621575092626) q[1];
h q[1];
cx q[7],q[1];
cx q[4],q[7];
rz(-1.9977215959937817) q[7];
h q[7];
rz(0.8480621575092626) q[7];
h q[7];
cx q[4],q[7];
cx q[9],q[8];
rz(-4.18026225750898) q[8];
h q[8];
rz(2.7083713851400066) q[8];
h q[8];
rz(3*pi) q[8];
cx q[9],q[8];
h q[9];
rz(3.5748139220395796) q[9];
h q[9];
cx q[8],q[2];
rz(-2.6709559895511186) q[2];
h q[2];
rz(0.8480621575092631) q[2];
h q[2];
cx q[8],q[2];
cx q[1],q[2];
rz(10.86394544881762) q[2];
cx q[1],q[2];
cx q[8],q[4];
rz(-2.344146266755612) q[4];
h q[4];
rz(0.8480621575092631) q[4];
h q[4];
cx q[8],q[4];
cx q[9],q[3];
rz(-3.1156747662027087) q[3];
h q[3];
rz(0.8480621575092631) q[3];
h q[3];
cx q[9],q[3];
cx q[0],q[3];
rz(11.281076550650738) q[3];
cx q[0],q[3];
cx q[0],q[6];
rz(10.812393218713382) q[6];
cx q[0],q[6];
cx q[2],q[6];
cx q[3],q[4];
rz(10.083508248375558) q[4];
cx q[3],q[4];
rz(4.765068633819417) q[6];
cx q[2],q[6];
cx q[7],q[0];
rz(-4.404124143343021) q[0];
h q[0];
rz(0.005591207485551752) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[9],q[5];
rz(-2.509745745087434) q[5];
h q[5];
rz(0.8480621575092626) q[5];
h q[5];
cx q[9],q[5];
cx q[1],q[5];
rz(10.593206548754843) q[5];
cx q[1],q[5];
cx q[5],q[6];
rz(-4.2606909037173395) q[6];
h q[6];
rz(0.005591207485551752) q[6];
h q[6];
rz(3*pi) q[6];
cx q[5],q[6];
cx q[7],q[1];
rz(1.5338571191995367) q[1];
h q[1];
rz(0.005591207485551752) q[1];
h q[1];
rz(3*pi) q[1];
cx q[7],q[1];
cx q[4],q[7];
rz(-4.552802907588486) q[7];
h q[7];
rz(0.005591207485551752) q[7];
h q[7];
rz(3*pi) q[7];
cx q[4],q[7];
cx q[9],q[8];
rz(-2.28800042934882) q[8];
h q[8];
rz(0.8480621575092626) q[8];
h q[8];
cx q[9],q[8];
h q[9];
rz(0.8480621575092626) q[9];
h q[9];
cx q[8],q[2];
rz(0.9650087104397036) q[2];
h q[2];
rz(0.005591207485551752) q[2];
h q[2];
rz(3*pi) q[2];
cx q[8],q[2];
cx q[1],q[2];
rz(10.195670887903727) q[2];
cx q[1],q[2];
cx q[8],q[4];
rz(1.336545823784225) q[4];
h q[4];
rz(0.005591207485551752) q[4];
h q[4];
rz(3*pi) q[4];
cx q[8],q[4];
cx q[9],q[3];
rz(0.45942543368197075) q[3];
h q[3];
rz(0.005591207485551752) q[3];
h q[3];
rz(3*pi) q[3];
cx q[9],q[3];
cx q[0],q[3];
rz(10.551947874851585) q[3];
cx q[0],q[3];
cx q[0],q[6];
rz(10.151639471566597) q[6];
cx q[0],q[6];
cx q[2],q[6];
cx q[3],q[4];
rz(9.529089534814496) q[4];
cx q[3],q[4];
rz(4.069905811377564) q[6];
cx q[2],q[6];
cx q[7],q[0];
rz(-1.994980935172452) q[0];
h q[0];
rz(1.2892372703678063) q[0];
h q[0];
cx q[7],q[0];
cx q[9],q[5];
rz(1.1482822934814179) q[5];
h q[5];
rz(0.005591207485551752) q[5];
h q[5];
rz(3*pi) q[5];
cx q[9],q[5];
cx q[1],q[5];
rz(9.96442934504094) q[5];
cx q[1],q[5];
cx q[5],q[6];
rz(-1.8724727783425332) q[6];
h q[6];
rz(1.2892372703678063) q[6];
h q[6];
cx q[5],q[6];
cx q[7],q[1];
rz(-2.2898241064560407) q[1];
h q[1];
rz(1.2892372703678063) q[1];
h q[1];
cx q[7],q[1];
cx q[4],q[7];
rz(-2.1219693613907813) q[7];
h q[7];
rz(1.2892372703678063) q[7];
h q[7];
cx q[4],q[7];
cx q[9],q[8];
rz(1.400375816900687) q[8];
h q[8];
rz(0.005591207485551752) q[8];
h q[8];
rz(3*pi) q[8];
cx q[9],q[8];
h q[9];
rz(6.277594099694035) q[9];
h q[9];
cx q[8],q[2];
rz(-2.7756847776272817) q[2];
h q[2];
rz(1.2892372703678063) q[2];
h q[2];
cx q[8],q[2];
cx q[8],q[4];
rz(-2.4583501969643207) q[4];
h q[4];
rz(1.2892372703678063) q[4];
h q[4];
cx q[8],q[4];
cx q[9],q[3];
rz(-3.2075098937667494) q[3];
h q[3];
rz(1.2892372703678063) q[3];
h q[3];
cx q[9],q[3];
cx q[9],q[5];
rz(-2.6191484760513726) q[5];
h q[5];
rz(1.2892372703678063) q[5];
h q[5];
cx q[9],q[5];
cx q[9],q[8];
rz(-2.403832186860309) q[8];
h q[8];
rz(1.2892372703678063) q[8];
h q[8];
cx q[9],q[8];
h q[9];
rz(1.2892372703678063) q[9];
h q[9];
