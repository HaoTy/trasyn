OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,6.73634121052393) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,5.333588759467526) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cx q[1],q[6];
u3(0,0,6.3010378375893445) q[6];
cx q[1],q[6];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cx q[0],q[9];
u3(0,0,5.539181290765397) q[9];
cx q[0],q[9];
cx q[4],q[9];
u3(0,0,6.860370766174393) q[9];
cx q[4],q[9];
cx q[7],q[9];
u3(2.9845616836271534,pi/2,-2.2244516893882635) q[9];
cx q[7],q[9];
u3(pi/2,0,pi) q[10];
cx q[3],q[10];
u3(0,0,6.954147285647365) q[10];
cx q[3],q[10];
cx q[4],q[10];
u3(0,0,6.235870980921351) q[10];
cx q[4],q[10];
u3(pi/2,0,pi) q[11];
cx q[5],q[11];
u3(0,0,5.919557548007915) q[11];
cx q[5],q[11];
cx q[8],q[11];
u3(0,0,5.849143121582702) q[11];
cx q[8],q[11];
u3(pi/2,0,pi) q[12];
cx q[5],q[12];
u3(0,0,5.946974373261295) q[12];
cx q[5],q[12];
cx q[6],q[12];
u3(0,0,7.835562010689205) q[12];
cx q[6],q[12];
u3(pi/2,0,pi) q[13];
cx q[13],q[3];
u3(2.9845616836271534,pi/2,-1.1039618413278065) q[3];
cx q[13],q[3];
cx q[8],q[13];
u3(0,0,6.90288585370372) q[13];
cx q[8],q[13];
u3(pi/2,0,pi) q[14];
cx q[14],q[5];
u3(2.9845616836271534,pi/2,-1.7927324233091286) q[5];
cx q[14],q[5];
cx q[14],q[8];
u3(2.9845616836271534,pi/2,-1.4257509000698492) q[8];
cx q[14],q[8];
u3(pi/2,0,pi) q[15];
cx q[15],q[11];
u3(2.9845616836271534,pi/2,-1.55119937715209) q[11];
cx q[15],q[11];
cx q[15],q[13];
u3(2.9845616836271534,pi/2,-2.2785238191307395) q[13];
cx q[15],q[13];
u3(pi/2,0,pi) q[16];
cx q[2],q[16];
u3(0,0,5.987035065608858) q[16];
cx q[2],q[16];
cx q[7],q[16];
u3(0,0,6.877002460063028) q[16];
cx q[7],q[16];
cx q[16],q[10];
u3(2.9845616836271534,pi/2,-1.8332768569992601) q[10];
cx q[16],q[10];
u3(3.2986236235524333,-pi/2,pi/2) q[16];
u3(pi/2,0,pi) q[17];
cx q[17],q[1];
u3(2.9845616836271534,pi/2,-2.2713520139103536) q[1];
cx q[17],q[1];
cx q[17],q[2];
u3(2.9845616836271534,pi/2,-2.2847107044556085) q[2];
cx q[17],q[2];
cx q[17],q[15];
u3(2.9845616836271534,pi/2,-1.5918400470580347) q[15];
cx q[17],q[15];
u3(3.2986236235524333,-pi/2,pi/2) q[17];
u3(pi/2,0,pi) q[18];
cx q[18],q[4];
u3(2.9845616836271534,pi/2,-1.5646311474393158) q[4];
cx q[18],q[4];
cx q[18],q[7];
u3(2.9845616836271534,pi/2,-1.7177254604749268) q[7];
cx q[18],q[7];
cx q[18],q[12];
u3(2.9845616836271534,pi/2,-1.0318117810803646) q[12];
cx q[18],q[12];
u3(3.2986236235524333,-pi/2,pi/2) q[18];
u3(pi/2,0,pi) q[19];
cx q[19],q[0];
u3(2.9845616836271534,pi/2,-2.5474317746236546) q[0];
cx q[19],q[0];
cx q[19],q[6];
u3(2.9845616836271534,pi/2,-2.904233119010634) q[6];
cx q[19],q[6];
cx q[19],q[14];
u3(2.9845616836271534,pi/2,-1.1892080859146241) q[14];
cx q[19],q[14];
u3(3.2986236235524333,-pi/2,pi/2) q[19];
