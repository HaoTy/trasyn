OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
cx q[0],q[1];
rz(5.50751229495052) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(5.599850626711198) q[4];
cx q[0],q[4];
cx q[24],q[0];
rz(6.502485813391658) q[0];
cx q[24],q[0];
cx q[1],q[9];
rz(4.529140161816187) q[9];
cx q[1],q[9];
cx q[13],q[1];
rz(5.424601887148495) q[1];
cx q[13],q[1];
cx q[2],q[18];
rz(5.741123983178179) q[18];
cx q[2],q[18];
cx q[2],q[20];
rz(6.7702464934439535) q[20];
cx q[2],q[20];
cx q[24],q[2];
rz(5.8819473614615205) q[2];
cx q[24],q[2];
cx q[3],q[12];
rz(5.348058384908489) q[12];
cx q[3],q[12];
cx q[3],q[17];
rz(6.095121945651807) q[17];
cx q[3],q[17];
cx q[25],q[3];
rz(5.525629193966661) q[3];
cx q[25],q[3];
cx q[4],q[11];
rz(5.331202583503114) q[11];
cx q[4],q[11];
cx q[16],q[4];
rz(5.309377712046763) q[4];
cx q[16],q[4];
cx q[5],q[6];
rz(6.340651297282658) q[6];
cx q[5],q[6];
cx q[5],q[13];
rz(5.404700048704024) q[13];
cx q[5],q[13];
cx q[18],q[5];
rz(5.748074764209021) q[5];
cx q[18],q[5];
cx q[6],q[9];
rz(5.429190084864144) q[9];
cx q[6],q[9];
cx q[25],q[6];
rz(7.167125999794021) q[6];
cx q[25],q[6];
cx q[7],q[12];
rz(6.341571782721764) q[12];
cx q[7],q[12];
cx q[7],q[13];
rz(5.789515559593311) q[13];
cx q[7],q[13];
cx q[14],q[7];
rz(5.091784927528082) q[7];
cx q[14],q[7];
cx q[8],q[10];
rz(5.2894606429065965) q[10];
cx q[8],q[10];
cx q[8],q[18];
rz(5.247110698030283) q[18];
cx q[8],q[18];
cx q[20],q[8];
rz(5.413046539555364) q[8];
cx q[20],q[8];
cx q[17],q[9];
rz(5.777818666300584) q[9];
cx q[17],q[9];
cx q[10],q[22];
rz(5.539517469067096) q[22];
cx q[10],q[22];
cx q[25],q[10];
rz(5.341125699715057) q[10];
cx q[25],q[10];
cx q[11],q[19];
rz(6.443740017889231) q[19];
cx q[11],q[19];
cx q[21],q[11];
rz(5.177948449095339) q[11];
cx q[21],q[11];
cx q[23],q[12];
rz(6.719174628777501) q[12];
cx q[23],q[12];
cx q[14],q[16];
rz(4.663437476347159) q[16];
cx q[14],q[16];
cx q[20],q[14];
rz(5.045461996388582) q[14];
cx q[20],q[14];
cx q[15],q[19];
rz(5.8181719321702525) q[19];
cx q[15],q[19];
cx q[15],q[21];
rz(6.557755538481873) q[21];
cx q[15],q[21];
cx q[23],q[15];
rz(4.4374026580629495) q[15];
cx q[23],q[15];
cx q[19],q[16];
rz(5.9480872900332455) q[16];
cx q[19],q[16];
cx q[22],q[17];
rz(6.314038888644286) q[17];
cx q[22],q[17];
cx q[22],q[21];
rz(4.694351616747142) q[21];
cx q[22],q[21];
cx q[24],q[23];
rz(5.758977416701755) q[23];
cx q[24],q[23];
rx(2.4516472871339614) q[0];
rx(2.4516472871339614) q[1];
rx(2.4516472871339614) q[2];
rx(2.4516472871339614) q[3];
rx(2.4516472871339614) q[4];
rx(2.4516472871339614) q[5];
rx(2.4516472871339614) q[6];
rx(2.4516472871339614) q[7];
rx(2.4516472871339614) q[8];
rx(2.4516472871339614) q[9];
rx(2.4516472871339614) q[10];
rx(2.4516472871339614) q[11];
rx(2.4516472871339614) q[12];
rx(2.4516472871339614) q[13];
rx(2.4516472871339614) q[14];
rx(2.4516472871339614) q[15];
rx(2.4516472871339614) q[16];
rx(2.4516472871339614) q[17];
rx(2.4516472871339614) q[18];
rx(2.4516472871339614) q[19];
rx(2.4516472871339614) q[20];
rx(2.4516472871339614) q[21];
rx(2.4516472871339614) q[22];
rx(2.4516472871339614) q[23];
rx(2.4516472871339614) q[24];
rx(2.4516472871339614) q[25];
