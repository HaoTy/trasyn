OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
h q[0];
h q[1];
h q[2];
cx q[0],q[2];
rz(0.5086348730341853) q[2];
cx q[0],q[2];
h q[3];
h q[4];
h q[5];
cx q[2],q[5];
rz(0.6039596941887285) q[5];
cx q[2],q[5];
cx q[3],q[5];
rz(0.5937045470973326) q[5];
cx q[3],q[5];
h q[6];
h q[7];
cx q[1],q[7];
rz(0.44268048602615706) q[7];
cx q[1],q[7];
h q[8];
cx q[4],q[8];
rz(0.5211676506542652) q[8];
cx q[4],q[8];
h q[9];
cx q[9],q[2];
rz(0.49083378784730947) q[2];
h q[2];
rz(2.1738824768425857) q[2];
h q[2];
cx q[9],q[2];
cx q[6],q[9];
rz(0.6255087438867355) q[9];
cx q[6],q[9];
h q[10];
cx q[0],q[10];
rz(0.4790687012397335) q[10];
cx q[0],q[10];
cx q[4],q[10];
rz(0.6217505971580012) q[10];
cx q[4],q[10];
h q[11];
cx q[11],q[4];
rz(0.5750340802138911) q[4];
h q[4];
rz(2.1738824768425857) q[4];
h q[4];
cx q[11],q[4];
cx q[8],q[11];
rz(0.4975988444449311) q[11];
cx q[8],q[11];
h q[12];
cx q[1],q[12];
rz(0.5988822256925777) q[12];
cx q[1],q[12];
h q[13];
cx q[3],q[13];
rz(0.5376762537972072) q[13];
cx q[3],q[13];
cx q[13],q[5];
rz(0.5929378019471265) q[5];
h q[5];
rz(2.1738824768425857) q[5];
h q[5];
cx q[13],q[5];
cx q[13],q[10];
rz(0.4983389008788315) q[10];
h q[10];
rz(2.1738824768425857) q[10];
h q[10];
cx q[13],q[10];
h q[13];
rz(2.1738824768425857) q[13];
h q[13];
h q[14];
cx q[14],q[0];
rz(0.5416447807460658) q[0];
h q[0];
rz(2.1738824768425857) q[0];
h q[0];
cx q[14],q[0];
cx q[14],q[3];
rz(0.5555422609884779) q[3];
h q[3];
rz(2.1738824768425857) q[3];
h q[3];
cx q[14],q[3];
cx q[7],q[14];
rz(0.5996160897794294) q[14];
h q[14];
rz(2.1738824768425857) q[14];
h q[14];
cx q[7],q[14];
h q[15];
cx q[6],q[15];
rz(0.4913819807360598) q[15];
cx q[6],q[15];
cx q[15],q[7];
rz(0.6690963393319169) q[7];
h q[7];
rz(2.1738824768425857) q[7];
h q[7];
cx q[15],q[7];
cx q[15],q[9];
rz(0.5692678797897797) q[9];
h q[9];
rz(2.1738824768425857) q[9];
h q[9];
cx q[15],q[9];
h q[15];
rz(2.1738824768425857) q[15];
h q[15];
h q[16];
cx q[16],q[6];
rz(0.4580509689928718) q[6];
h q[6];
rz(2.1738824768425857) q[6];
h q[6];
cx q[16],q[6];
cx q[16],q[8];
rz(0.4586381277532219) q[8];
h q[8];
rz(2.1738824768425857) q[8];
h q[8];
cx q[16],q[8];
cx q[12],q[16];
rz(0.5890292757322433) q[16];
h q[16];
rz(2.1738824768425857) q[16];
h q[16];
cx q[12],q[16];
h q[17];
cx q[17],q[1];
rz(0.47724070380584216) q[1];
h q[1];
rz(2.1738824768425857) q[1];
h q[1];
cx q[17],q[1];
cx q[17],q[11];
rz(0.5601163188934191) q[11];
h q[11];
rz(2.1738824768425857) q[11];
h q[11];
cx q[17],q[11];
cx q[17],q[12];
rz(0.5596757851236855) q[12];
h q[12];
rz(2.1738824768425857) q[12];
h q[12];
cx q[17],q[12];
h q[17];
rz(2.1738824768425857) q[17];
h q[17];
