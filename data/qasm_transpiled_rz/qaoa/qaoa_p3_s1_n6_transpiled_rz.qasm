OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3423163608908018) q[1];
cx q[0],q[1];
h q[2];
cx q[1],q[2];
rz(0.3221911585416071) q[2];
cx q[1],q[2];
h q[3];
cx q[0],q[3];
rz(0.32609336806008293) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(0.3356179246615296) q[3];
cx q[2],q[3];
h q[4];
cx q[4],q[0];
rz(0.36133014453171786) q[0];
h q[0];
rz(2.526252418241935) q[0];
h q[0];
cx q[4],q[0];
cx q[4],q[2];
rz(0.3003443308078184) q[2];
h q[2];
rz(2.526252418241935) q[2];
h q[2];
cx q[4],q[2];
h q[5];
cx q[5],q[1];
rz(0.3365562242342257) q[1];
h q[1];
rz(2.526252418241935) q[1];
h q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(6.395139569309023) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(6.3885576397356125) q[2];
cx q[1],q[2];
cx q[5],q[3];
rz(0.3440110287633722) q[3];
h q[3];
rz(2.526252418241935) q[3];
h q[3];
cx q[5],q[3];
cx q[0],q[3];
rz(6.389833853896626) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(0.1097635445034437) q[3];
cx q[2],q[3];
cx q[5],q[4];
rz(0.38114312540483564) q[4];
h q[4];
rz(2.526252418241935) q[4];
h q[4];
cx q[5],q[4];
h q[5];
rz(2.526252418241935) q[5];
h q[5];
cx q[4],q[0];
rz(-3.0234199503894477) q[0];
h q[0];
rz(1.0199549368252177) q[0];
h q[0];
rz(3*pi) q[0];
cx q[4],q[0];
cx q[4],q[2];
rz(-3.0433653066618005) q[2];
h q[2];
rz(1.0199549368252177) q[2];
h q[2];
rz(3*pi) q[2];
cx q[4],q[2];
cx q[5],q[1];
rz(-3.03152223904355) q[1];
h q[1];
rz(1.0199549368252177) q[1];
h q[1];
rz(3*pi) q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(8.830033541365285) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(8.680301182875919) q[2];
cx q[1],q[2];
cx q[5],q[3];
rz(-3.029084151836126) q[3];
h q[3];
rz(1.0199549368252177) q[3];
h q[3];
rz(3*pi) q[3];
cx q[5],q[3];
cx q[0],q[3];
rz(8.709333787089435) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(2.497011584725145) q[3];
cx q[2],q[3];
cx q[5],q[4];
rz(-3.0169401325727874) q[4];
h q[4];
rz(1.0199549368252177) q[4];
h q[4];
rz(3*pi) q[4];
cx q[5],q[4];
h q[5];
rz(5.263230370354368) q[5];
h q[5];
cx q[4],q[0];
rz(-0.45328106321206363) q[0];
h q[0];
rz(0.029725617570441454) q[0];
h q[0];
rz(3*pi) q[0];
cx q[4],q[0];
cx q[4],q[2];
rz(-0.9070181022156927) q[2];
h q[2];
rz(0.029725617570441454) q[2];
h q[2];
rz(3*pi) q[2];
cx q[4],q[2];
cx q[5],q[1];
rz(-0.6376000802737432) q[1];
h q[1];
rz(0.029725617570441454) q[1];
h q[1];
rz(3*pi) q[1];
cx q[5],q[1];
cx q[5],q[3];
rz(-0.5821360186032827) q[3];
h q[3];
rz(0.029725617570441454) q[3];
h q[3];
rz(3*pi) q[3];
cx q[5],q[3];
cx q[5],q[4];
rz(-0.3058716457382875) q[4];
h q[4];
rz(0.029725617570441454) q[4];
h q[4];
rz(3*pi) q[4];
cx q[5],q[4];
h q[5];
rz(6.253459689609144) q[5];
h q[5];
