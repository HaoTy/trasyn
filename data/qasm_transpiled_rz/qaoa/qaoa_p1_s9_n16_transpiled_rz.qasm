OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
h q[0];
h q[1];
h q[2];
cx q[0],q[2];
rz(1.0763534977245135) q[2];
cx q[0],q[2];
h q[3];
cx q[2],q[3];
rz(1.3813632527408801) q[3];
cx q[2],q[3];
h q[4];
cx q[3],q[4];
rz(1.1640416538963947) q[4];
cx q[3],q[4];
h q[5];
cx q[0],q[5];
rz(1.4638649081416601) q[5];
cx q[0],q[5];
h q[6];
h q[7];
cx q[1],q[7];
rz(1.373253935834149) q[7];
cx q[1],q[7];
h q[8];
cx q[5],q[8];
rz(1.1633395804069286) q[8];
cx q[5],q[8];
h q[9];
cx q[1],q[9];
rz(1.2434874614398084) q[9];
cx q[1],q[9];
cx q[6],q[9];
rz(1.2570477235721929) q[9];
cx q[6],q[9];
cx q[7],q[9];
rz(-1.9158927194336592) q[9];
h q[9];
rz(0.5485422559694335) q[9];
h q[9];
rz(3*pi) q[9];
cx q[7],q[9];
h q[10];
cx q[8],q[10];
rz(1.5034165158451762) q[10];
cx q[8],q[10];
h q[11];
cx q[4],q[11];
rz(1.3365476248376307) q[11];
cx q[4],q[11];
cx q[11],q[5];
rz(-1.8870658586564812) q[5];
h q[5];
rz(0.5485422559694335) q[5];
h q[5];
rz(3*pi) q[5];
cx q[11],q[5];
cx q[11],q[8];
rz(-2.0266383956911262) q[8];
h q[8];
rz(0.5485422559694335) q[8];
h q[8];
rz(3*pi) q[8];
cx q[11],q[8];
h q[11];
rz(5.734643051210153) q[11];
h q[11];
h q[12];
cx q[12],q[2];
rz(-1.8202765617266814) q[2];
h q[2];
rz(0.5485422559694335) q[2];
h q[2];
rz(3*pi) q[2];
cx q[12],q[2];
cx q[10],q[12];
rz(1.2204397496494237) q[12];
cx q[10],q[12];
h q[13];
cx q[13],q[3];
rz(-2.13966214419406) q[3];
h q[3];
rz(0.5485422559694335) q[3];
h q[3];
rz(3*pi) q[3];
cx q[13],q[3];
cx q[13],q[4];
rz(-1.8890570276716012) q[4];
h q[4];
rz(0.548542255969434) q[4];
h q[4];
rz(3*pi) q[4];
cx q[13],q[4];
cx q[13],q[7];
rz(-2.1822649597198063) q[7];
h q[7];
rz(0.5485422559694335) q[7];
h q[7];
rz(3*pi) q[7];
cx q[13],q[7];
h q[13];
rz(5.734643051210153) q[13];
h q[13];
h q[14];
cx q[6],q[14];
rz(1.4376755049752643) q[14];
cx q[6],q[14];
cx q[14],q[10];
rz(-1.9927758011328072) q[10];
h q[10];
rz(0.5485422559694335) q[10];
h q[10];
rz(3*pi) q[10];
cx q[14],q[10];
cx q[14],q[12];
rz(-1.8300901858010237) q[12];
h q[12];
rz(0.5485422559694335) q[12];
h q[12];
rz(3*pi) q[12];
cx q[14],q[12];
h q[14];
rz(5.734643051210153) q[14];
h q[14];
h q[15];
cx q[15],q[0];
rz(-1.6329611872779788) q[0];
h q[0];
rz(0.5485422559694335) q[0];
h q[0];
rz(3*pi) q[0];
cx q[15],q[0];
cx q[15],q[1];
rz(-1.7887642739817915) q[1];
h q[1];
rz(0.5485422559694335) q[1];
h q[1];
rz(3*pi) q[1];
cx q[15],q[1];
cx q[15],q[6];
rz(-1.7551338757157637) q[6];
h q[6];
rz(0.5485422559694335) q[6];
h q[6];
rz(3*pi) q[6];
cx q[15],q[6];
h q[15];
rz(5.734643051210153) q[15];
h q[15];
