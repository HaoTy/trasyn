OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
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
cx q[0],q[1];
rz(6.73634121052393) q[1];
cx q[0],q[1];
cx q[0],q[9];
rz(5.539181290765397) q[9];
cx q[0],q[9];
cx q[19],q[0];
rz(5.306549859350828) q[0];
cx q[19],q[0];
cx q[1],q[6];
rz(6.3010378375893445) q[6];
cx q[1],q[6];
cx q[17],q[1];
rz(5.58262962006413) q[1];
cx q[17],q[1];
cx q[2],q[3];
rz(5.333588759467526) q[3];
cx q[2],q[3];
cx q[2],q[16];
rz(5.987035065608858) q[16];
cx q[2],q[16];
cx q[17],q[2];
rz(5.569270929518875) q[2];
cx q[17],q[2];
cx q[3],q[10];
rz(6.954147285647365) q[10];
cx q[3],q[10];
cx q[13],q[3];
rz(6.750019792646676) q[3];
cx q[13],q[3];
cx q[4],q[9];
rz(6.860370766174393) q[9];
cx q[4],q[9];
cx q[4],q[10];
rz(6.235870980921351) q[10];
cx q[4],q[10];
cx q[18],q[4];
rz(6.2893504865351675) q[4];
cx q[18],q[4];
cx q[5],q[11];
rz(5.919557548007915) q[11];
cx q[5],q[11];
cx q[5],q[12];
rz(5.946974373261295) q[12];
cx q[5],q[12];
cx q[14],q[5];
rz(6.061249210665355) q[5];
cx q[14],q[5];
cx q[6],q[12];
rz(7.835562010689205) q[12];
cx q[6],q[12];
cx q[19],q[6];
rz(4.949748514963849) q[6];
cx q[19],q[6];
cx q[7],q[9];
rz(5.62952994458622) q[9];
cx q[7],q[9];
cx q[7],q[16];
rz(6.877002460063028) q[16];
cx q[7],q[16];
cx q[18],q[7];
rz(6.136256173499556) q[7];
cx q[18],q[7];
cx q[8],q[11];
rz(5.849143121582702) q[11];
cx q[8],q[11];
cx q[8],q[13];
rz(6.90288585370372) q[13];
cx q[8],q[13];
cx q[14],q[8];
rz(6.428230733904634) q[8];
cx q[14],q[8];
cx q[16],q[10];
rz(6.020704776975223) q[10];
cx q[16],q[10];
cx q[15],q[11];
rz(6.302782256822393) q[11];
cx q[15],q[11];
cx q[18],q[12];
rz(6.822169852894119) q[12];
cx q[18],q[12];
cx q[15],q[13];
rz(5.575457814843744) q[13];
cx q[15],q[13];
cx q[19],q[14];
rz(6.664773548059859) q[14];
cx q[19],q[14];
cx q[17],q[15];
rz(6.2621415869164485) q[15];
cx q[17],q[15];
rx(3.2986236235524333) q[0];
rx(3.2986236235524333) q[1];
rx(3.2986236235524333) q[2];
rx(3.2986236235524333) q[3];
rx(3.2986236235524333) q[4];
rx(3.2986236235524333) q[5];
rx(3.2986236235524333) q[6];
rx(3.2986236235524333) q[7];
rx(3.2986236235524333) q[8];
rx(3.2986236235524333) q[9];
rx(3.2986236235524333) q[10];
rx(3.2986236235524333) q[11];
rx(3.2986236235524333) q[12];
rx(3.2986236235524333) q[13];
rx(3.2986236235524333) q[14];
rx(3.2986236235524333) q[15];
rx(3.2986236235524333) q[16];
rx(3.2986236235524333) q[17];
rx(3.2986236235524333) q[18];
rx(3.2986236235524333) q[19];
