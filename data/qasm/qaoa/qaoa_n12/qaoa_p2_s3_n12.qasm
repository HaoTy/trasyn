OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
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
cx q[0],q[4];
rz(5.485797239924479) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(5.5772646286767165) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(4.680148172184072) q[0];
cx q[6],q[0];
cx q[1],q[3];
rz(5.854271268286262) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(5.2098795716639135) q[7];
cx q[1],q[7];
cx q[8],q[1];
rz(5.018611231048774) q[1];
cx q[8],q[1];
cx q[2],q[5];
rz(4.961856374811063) q[5];
cx q[2],q[5];
cx q[2],q[6];
rz(5.632365900244164) q[6];
cx q[2],q[6];
cx q[10],q[2];
rz(4.45670432107095) q[2];
cx q[10],q[2];
cx q[3],q[6];
rz(6.172569269767606) q[6];
cx q[3],q[6];
cx q[11],q[3];
rz(4.399290455966888) q[3];
cx q[11],q[3];
cx q[4],q[5];
rz(4.286427072121192) q[5];
cx q[4],q[5];
cx q[11],q[4];
rz(5.9178835362835835) q[4];
cx q[11],q[4];
cx q[7],q[9];
rz(5.185129740793035) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(4.684082397190024) q[7];
cx q[10],q[7];
cx q[8],q[9];
rz(4.656916662613918) q[9];
cx q[8],q[9];
cx q[11],q[8];
rz(5.427108369336772) q[8];
cx q[11],q[8];
cx q[10],q[9];
rz(4.152956660847818) q[9];
cx q[10],q[9];
rx(1.2609752482504697) q[0];
rx(1.2609752482504697) q[1];
rx(1.2609752482504697) q[2];
rx(1.2609752482504697) q[3];
rx(1.2609752482504697) q[4];
rx(1.2609752482504697) q[5];
rx(1.2609752482504697) q[6];
rx(1.2609752482504697) q[7];
rx(1.2609752482504697) q[8];
rx(1.2609752482504697) q[9];
rx(1.2609752482504697) q[10];
rx(1.2609752482504697) q[11];
cx q[0],q[4];
rz(5.809573559170786) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(5.906439429340773) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(4.956374412864078) q[0];
cx q[6],q[0];
cx q[1],q[3];
rz(6.199795231388704) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(5.517371000467267) q[7];
cx q[1],q[7];
cx q[8],q[1];
rz(5.314813843185332) q[1];
cx q[8],q[1];
cx q[2],q[5];
rz(5.254709268092126) q[5];
cx q[2],q[5];
cx q[2],q[6];
rz(5.964792823820103) q[6];
cx q[2],q[6];
cx q[10],q[2];
rz(4.719742719673033) q[2];
cx q[10],q[2];
cx q[3],q[6];
rz(6.536879445855291) q[6];
cx q[3],q[6];
cx q[11],q[3];
rz(4.658940240461654) q[3];
cx q[11],q[3];
cx q[4],q[5];
rz(4.53941556575867) q[5];
cx q[4],q[5];
cx q[11],q[4];
rz(6.267161948391392) q[4];
cx q[11],q[4];
cx q[7],q[9];
rz(5.491160413977675) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(4.9605408391045644) q[7];
cx q[10],q[7];
cx q[8],q[9];
rz(4.9317717602621665) q[9];
cx q[8],q[9];
cx q[11],q[8];
rz(5.747420822590857) q[8];
cx q[11],q[8];
cx q[10],q[9];
rz(4.398067619716807) q[9];
cx q[10],q[9];
rx(4.063005906916554) q[0];
rx(4.063005906916554) q[1];
rx(4.063005906916554) q[2];
rx(4.063005906916554) q[3];
rx(4.063005906916554) q[4];
rx(4.063005906916554) q[5];
rx(4.063005906916554) q[6];
rx(4.063005906916554) q[7];
rx(4.063005906916554) q[8];
rx(4.063005906916554) q[9];
rx(4.063005906916554) q[10];
rx(4.063005906916554) q[11];
