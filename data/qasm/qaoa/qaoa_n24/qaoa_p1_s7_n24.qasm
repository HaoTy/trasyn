OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
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
cx q[0],q[13];
rz(5.79328829569698) q[13];
cx q[0],q[13];
cx q[0],q[16];
rz(6.599556515066336) q[16];
cx q[0],q[16];
cx q[21],q[0];
rz(5.778995189539948) q[0];
cx q[21],q[0];
cx q[1],q[14];
rz(4.773863622886403) q[14];
cx q[1],q[14];
cx q[1],q[17];
rz(6.5381105918200575) q[17];
cx q[1],q[17];
cx q[21],q[1];
rz(4.898526066571776) q[1];
cx q[21],q[1];
cx q[2],q[3];
rz(5.337042662680175) q[3];
cx q[2],q[3];
cx q[2],q[4];
rz(5.6976092759321295) q[4];
cx q[2],q[4];
cx q[6],q[2];
rz(4.772443641661272) q[2];
cx q[6],q[2];
cx q[3],q[4];
rz(4.911427356130299) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(5.757567297482252) q[3];
cx q[5],q[3];
cx q[15],q[4];
rz(6.1151428722095895) q[4];
cx q[15],q[4];
cx q[5],q[15];
rz(5.3671627433677305) q[15];
cx q[5],q[15];
cx q[20],q[5];
rz(5.402304509132846) q[5];
cx q[20],q[5];
cx q[6],q[7];
rz(6.206802201218129) q[7];
cx q[6],q[7];
cx q[18],q[6];
rz(4.612784927188559) q[6];
cx q[18],q[6];
cx q[7],q[14];
rz(5.942802125575939) q[14];
cx q[7],q[14];
cx q[22],q[7];
rz(4.807952084491109) q[7];
cx q[22],q[7];
cx q[8],q[10];
rz(4.9600960574096575) q[10];
cx q[8],q[10];
cx q[8],q[18];
rz(5.279150146019257) q[18];
cx q[8],q[18];
cx q[20],q[8];
rz(4.794669697193202) q[8];
cx q[20],q[8];
cx q[9],q[10];
rz(4.855001125402174) q[10];
cx q[9],q[10];
cx q[9],q[12];
rz(4.836674566174468) q[12];
cx q[9],q[12];
cx q[13],q[9];
rz(4.3900191257157735) q[9];
cx q[13],q[9];
cx q[23],q[10];
rz(5.372013922648515) q[10];
cx q[23],q[10];
cx q[11],q[12];
rz(5.968381293896766) q[12];
cx q[11],q[12];
cx q[11],q[19];
rz(4.823152967506987) q[19];
cx q[11],q[19];
cx q[23],q[11];
rz(4.626905242108617) q[11];
cx q[23],q[11];
cx q[16],q[12];
rz(5.09955890364497) q[12];
cx q[16],q[12];
cx q[15],q[13];
rz(4.9044518688670875) q[13];
cx q[15],q[13];
cx q[16],q[14];
rz(5.271523831860855) q[14];
cx q[16],q[14];
cx q[17],q[19];
rz(5.651411734452085) q[19];
cx q[17],q[19];
cx q[22],q[17];
rz(4.606093315053986) q[17];
cx q[22],q[17];
cx q[20],q[18];
rz(5.1748628578310525) q[18];
cx q[20],q[18];
cx q[23],q[19];
rz(5.273664776612483) q[19];
cx q[23],q[19];
cx q[22],q[21];
rz(5.3837476116716525) q[21];
cx q[22],q[21];
rx(0.21502288168793493) q[0];
rx(0.21502288168793493) q[1];
rx(0.21502288168793493) q[2];
rx(0.21502288168793493) q[3];
rx(0.21502288168793493) q[4];
rx(0.21502288168793493) q[5];
rx(0.21502288168793493) q[6];
rx(0.21502288168793493) q[7];
rx(0.21502288168793493) q[8];
rx(0.21502288168793493) q[9];
rx(0.21502288168793493) q[10];
rx(0.21502288168793493) q[11];
rx(0.21502288168793493) q[12];
rx(0.21502288168793493) q[13];
rx(0.21502288168793493) q[14];
rx(0.21502288168793493) q[15];
rx(0.21502288168793493) q[16];
rx(0.21502288168793493) q[17];
rx(0.21502288168793493) q[18];
rx(0.21502288168793493) q[19];
rx(0.21502288168793493) q[20];
rx(0.21502288168793493) q[21];
rx(0.21502288168793493) q[22];
rx(0.21502288168793493) q[23];
