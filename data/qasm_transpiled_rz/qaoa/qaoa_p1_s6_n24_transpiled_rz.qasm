OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
cx q[2],q[4];
rz(0.71802337000468) q[4];
cx q[2],q[4];
h q[5];
cx q[1],q[5];
rz(0.4534581887170864) q[5];
cx q[1],q[5];
cx q[3],q[5];
rz(0.6580525442570397) q[5];
cx q[3],q[5];
h q[6];
cx q[2],q[6];
rz(0.5960212436680811) q[6];
cx q[2],q[6];
h q[7];
h q[8];
cx q[0],q[8];
rz(0.7242277184456561) q[8];
cx q[0],q[8];
h q[9];
cx q[0],q[9];
rz(0.7183131498197143) q[9];
cx q[0],q[9];
h q[10];
cx q[8],q[10];
rz(0.7063195026209048) q[10];
cx q[8],q[10];
h q[11];
cx q[7],q[11];
rz(0.7308518912948868) q[11];
cx q[7],q[11];
h q[12];
cx q[12],q[0];
rz(0.6410578824087896) q[0];
h q[0];
rz(1.4171344469393974) q[0];
h q[0];
cx q[12],q[0];
cx q[3],q[12];
rz(0.6262800712018958) q[12];
cx q[3],q[12];
cx q[6],q[12];
rz(0.5794715422357619) q[12];
h q[12];
rz(1.4171344469393974) q[12];
h q[12];
cx q[6],q[12];
h q[13];
cx q[1],q[13];
rz(0.7339476577418169) q[13];
cx q[1],q[13];
cx q[4],q[13];
rz(0.648088041938223) q[13];
cx q[4],q[13];
h q[14];
cx q[7],q[14];
rz(0.6165184194254634) q[14];
cx q[7],q[14];
cx q[14],q[8];
rz(0.6613951073158857) q[8];
h q[8];
rz(1.4171344469393974) q[8];
h q[8];
cx q[14],q[8];
h q[15];
cx q[9],q[15];
rz(0.6142348963442995) q[15];
cx q[9],q[15];
cx q[11],q[15];
rz(0.5956986763645656) q[15];
cx q[11],q[15];
h q[16];
cx q[16],q[1];
rz(0.6564996007561259) q[1];
h q[1];
rz(1.4171344469393974) q[1];
h q[1];
cx q[16],q[1];
h q[17];
cx q[17],q[2];
rz(0.6018953981668673) q[2];
h q[2];
rz(1.4171344469393974) q[2];
h q[2];
cx q[17],q[2];
cx q[17],q[7];
rz(0.6740544225499852) q[7];
h q[7];
rz(1.4171344469393974) q[7];
h q[7];
cx q[17],q[7];
cx q[17],q[14];
rz(0.7185364879392679) q[14];
h q[14];
rz(1.4171344469393974) q[14];
h q[14];
cx q[17],q[14];
h q[17];
rz(1.4171344469393974) q[17];
h q[17];
h q[18];
cx q[18],q[11];
rz(0.664455758551167) q[11];
h q[11];
rz(1.4171344469393974) q[11];
h q[11];
cx q[18],q[11];
cx q[16],q[18];
rz(0.7232139975618688) q[18];
cx q[16],q[18];
h q[19];
cx q[19],q[5];
rz(0.6795756553381933) q[5];
h q[5];
rz(1.4171344469393974) q[5];
h q[5];
cx q[19],q[5];
cx q[10],q[19];
rz(0.7533293034773552) q[19];
cx q[10],q[19];
h q[20];
cx q[20],q[4];
rz(0.7391994987974719) q[4];
h q[4];
rz(1.4171344469393974) q[4];
h q[4];
cx q[20],q[4];
cx q[20],q[9];
rz(0.7783725403614188) q[9];
h q[9];
rz(1.4171344469393974) q[9];
h q[9];
cx q[20],q[9];
cx q[20],q[19];
rz(0.7225895566961373) q[19];
h q[19];
rz(1.4171344469393974) q[19];
h q[19];
cx q[20],q[19];
h q[20];
rz(1.4171344469393974) q[20];
h q[20];
h q[21];
cx q[21],q[3];
rz(0.75157832942525) q[3];
h q[3];
rz(1.4171344469393974) q[3];
h q[3];
cx q[21],q[3];
cx q[21],q[6];
rz(0.7496981549518367) q[6];
h q[6];
rz(1.4171344469393974) q[6];
h q[6];
cx q[21],q[6];
cx q[21],q[16];
rz(0.6891333866608118) q[16];
h q[16];
rz(1.4171344469393974) q[16];
h q[16];
cx q[21],q[16];
h q[21];
rz(1.4171344469393974) q[21];
h q[21];
h q[22];
cx q[22],q[10];
rz(0.6384942001276341) q[10];
h q[10];
rz(1.4171344469393974) q[10];
h q[10];
cx q[22],q[10];
cx q[22],q[13];
rz(0.7927261161887333) q[13];
h q[13];
rz(1.4171344469393974) q[13];
h q[13];
cx q[22],q[13];
h q[23];
cx q[23],q[15];
rz(0.8026879570299652) q[15];
h q[15];
rz(1.4171344469393974) q[15];
h q[15];
cx q[23],q[15];
cx q[23],q[18];
rz(0.6635211746232255) q[18];
h q[18];
rz(1.4171344469393974) q[18];
h q[18];
cx q[23],q[18];
cx q[23],q[22];
rz(0.6638068747729413) q[22];
h q[22];
rz(1.4171344469393974) q[22];
h q[22];
cx q[23],q[22];
h q[23];
rz(1.4171344469393974) q[23];
h q[23];
