OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
h q[0];
h q[1];
cx q[0],q[1];
rz(2.976818565429584) q[1];
cx q[0],q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
cx q[5],q[6];
rz(3.2305408951215573) q[6];
cx q[5],q[6];
h q[7];
cx q[5],q[7];
rz(3.0697021203711223) q[7];
cx q[5],q[7];
h q[8];
cx q[6],q[8];
rz(3.350882374088695) q[8];
cx q[6],q[8];
h q[9];
cx q[1],q[9];
rz(3.1361827509180853) q[9];
cx q[1],q[9];
cx q[2],q[9];
rz(3.275000804701063) q[9];
cx q[2],q[9];
cx q[9],q[5];
rz(-2.8175501644886145) q[5];
h q[5];
rz(2.0615726274791566) q[5];
h q[5];
cx q[9],q[5];
h q[9];
rz(2.0615726274791566) q[9];
h q[9];
h q[10];
cx q[3],q[10];
rz(3.3397861260672186) q[10];
cx q[3],q[10];
h q[11];
cx q[10],q[11];
rz(3.2721203214918635) q[11];
cx q[10],q[11];
h q[12];
cx q[4],q[12];
rz(3.700368988172434) q[12];
cx q[4],q[12];
h q[13];
cx q[13],q[6];
rz(-2.6919982386817987) q[6];
h q[6];
rz(2.0615726274791566) q[6];
h q[6];
cx q[13],q[6];
h q[14];
cx q[0],q[14];
rz(3.1861270492223164) q[14];
cx q[0],q[14];
cx q[14],q[10];
rz(-3.4095673422374873) q[10];
h q[10];
rz(2.0615726274791566) q[10];
h q[10];
cx q[14],q[10];
h q[15];
h q[16];
cx q[3],q[16];
rz(3.0814198976520024) q[16];
cx q[3],q[16];
cx q[15],q[16];
rz(2.896541900427515) q[16];
cx q[15],q[16];
h q[17];
cx q[17],q[0];
rz(-2.930424522633409) q[0];
h q[0];
rz(2.0615726274791566) q[0];
h q[0];
cx q[17],q[0];
cx q[7],q[17];
rz(2.9272098326487574) q[17];
cx q[7],q[17];
cx q[12],q[17];
rz(-2.8915661741430867) q[17];
h q[17];
rz(2.0615726274791566) q[17];
h q[17];
cx q[12],q[17];
h q[18];
cx q[18],q[3];
rz(-2.758315305732064) q[3];
h q[3];
rz(2.0615726274791566) q[3];
h q[3];
cx q[18],q[3];
h q[19];
cx q[4],q[19];
rz(3.750851566758039) q[19];
cx q[4],q[19];
cx q[8],q[19];
rz(3.543104363522507) q[19];
cx q[8],q[19];
cx q[13],q[19];
rz(-3.4949322236504354) q[19];
h q[19];
rz(2.0615726274791566) q[19];
h q[19];
cx q[13],q[19];
h q[20];
cx q[20],q[4];
rz(-3.2061805117271938) q[4];
h q[4];
rz(2.0615726274791566) q[4];
h q[4];
cx q[20],q[4];
cx q[20],q[7];
rz(-3.344653648059368) q[7];
h q[7];
rz(2.0615726274791566) q[7];
h q[7];
cx q[20],q[7];
cx q[20],q[8];
rz(-2.961426887071199) q[8];
h q[8];
rz(2.0615726274791566) q[8];
h q[8];
cx q[20],q[8];
h q[20];
rz(2.0615726274791566) q[20];
h q[20];
h q[21];
cx q[21],q[12];
rz(-3.397704343439427) q[12];
h q[12];
rz(2.0615726274791566) q[12];
h q[12];
cx q[21],q[12];
cx q[15],q[21];
rz(3.041099927634911) q[21];
cx q[15],q[21];
h q[22];
cx q[11],q[22];
rz(3.436885746446289) q[22];
cx q[11],q[22];
cx q[22],q[14];
rz(-2.5164825600543237) q[14];
h q[14];
rz(2.0615726274791566) q[14];
h q[14];
cx q[22],q[14];
cx q[18],q[22];
rz(-2.7132729529674995) q[22];
h q[22];
rz(2.0615726274791566) q[22];
h q[22];
cx q[18],q[22];
h q[23];
cx q[2],q[23];
rz(2.8458156094807565) q[23];
cx q[2],q[23];
cx q[23],q[16];
rz(-3.220660631928263) q[16];
h q[16];
rz(2.0615726274791566) q[16];
h q[16];
cx q[23],q[16];
cx q[23],q[18];
rz(-2.9932694116447713) q[18];
h q[18];
rz(2.0615726274791566) q[18];
h q[18];
cx q[23],q[18];
h q[23];
rz(2.0615726274791566) q[23];
h q[23];
h q[24];
cx q[24],q[1];
rz(-3.2952381223753386) q[1];
h q[1];
rz(2.0615726274791566) q[1];
h q[1];
cx q[24],q[1];
cx q[24],q[2];
rz(-2.889903279095302) q[2];
h q[2];
rz(2.0615726274791575) q[2];
h q[2];
cx q[24],q[2];
cx q[24],q[11];
rz(-2.7586225665830266) q[11];
h q[11];
rz(2.0615726274791566) q[11];
h q[11];
cx q[24],q[11];
h q[24];
rz(2.0615726274791566) q[24];
h q[24];
h q[25];
cx q[25],q[13];
rz(-3.0811219612824505) q[13];
h q[13];
rz(2.0615726274791566) q[13];
h q[13];
cx q[25],q[13];
cx q[25],q[15];
rz(-2.730785422650559) q[15];
h q[15];
rz(2.0615726274791566) q[15];
h q[15];
cx q[25],q[15];
cx q[25],q[21];
rz(-3.0082178251501315) q[21];
h q[21];
rz(2.0615726274791566) q[21];
h q[21];
cx q[25],q[21];
h q[25];
rz(2.0615726274791566) q[25];
h q[25];
