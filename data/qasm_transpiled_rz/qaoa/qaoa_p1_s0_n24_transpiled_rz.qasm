OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[1],q[3];
rz(0.1514238871059777) q[3];
cx q[1],q[3];
h q[4];
cx q[2],q[4];
rz(0.18414052034986267) q[4];
cx q[2],q[4];
h q[5];
h q[6];
h q[7];
cx q[4],q[7];
rz(0.17568625208786692) q[7];
cx q[4],q[7];
cx q[6],q[7];
rz(0.14021899483741387) q[7];
cx q[6],q[7];
h q[8];
cx q[5],q[8];
rz(0.19911692114533952) q[8];
cx q[5],q[8];
h q[9];
h q[10];
cx q[3],q[10];
rz(0.18613341154404683) q[10];
cx q[3],q[10];
cx q[10],q[7];
rz(0.17696508079536954) q[7];
h q[7];
rz(2.8489343524921047) q[7];
h q[7];
cx q[10],q[7];
h q[11];
cx q[2],q[11];
rz(0.15371959880254843) q[11];
cx q[2],q[11];
cx q[9],q[11];
rz(0.18122985749178994) q[11];
cx q[9],q[11];
h q[12];
cx q[1],q[12];
rz(0.15706912222910854) q[12];
cx q[1],q[12];
cx q[9],q[12];
rz(0.14615976848351775) q[12];
cx q[9],q[12];
h q[13];
cx q[0],q[13];
rz(0.13477616012994806) q[13];
cx q[0],q[13];
h q[14];
cx q[6],q[14];
rz(0.16449813590043233) q[14];
cx q[6],q[14];
cx q[14],q[9];
rz(0.16524644035448866) q[9];
h q[9];
rz(2.8489343524921047) q[9];
h q[9];
cx q[14],q[9];
h q[15];
cx q[0],q[15];
rz(0.1608082554138099) q[15];
cx q[0],q[15];
cx q[15],q[6];
rz(0.17842051148893923) q[6];
h q[6];
rz(2.8489343524921047) q[6];
h q[6];
cx q[15],q[6];
h q[16];
cx q[16],q[0];
rz(0.15606894496269597) q[0];
h q[0];
rz(2.8489343524921047) q[0];
h q[0];
cx q[16],q[0];
cx q[16],q[10];
rz(0.15120438284529136) q[10];
h q[10];
rz(2.8489343524921047) q[10];
h q[10];
cx q[16],q[10];
h q[17];
cx q[17],q[11];
rz(0.18235124442616524) q[11];
h q[11];
rz(2.8489343524921047) q[11];
h q[11];
cx q[17],q[11];
cx q[17],q[16];
rz(0.17968878016742895) q[16];
h q[16];
rz(2.8489343524921047) q[16];
h q[16];
cx q[17],q[16];
h q[18];
cx q[5],q[18];
rz(0.16700538122642736) q[18];
cx q[5],q[18];
cx q[8],q[18];
rz(0.16196375041307776) q[18];
cx q[8],q[18];
cx q[18],q[15];
rz(0.16817567095545272) q[15];
h q[15];
rz(2.8489343524921047) q[15];
h q[15];
cx q[18],q[15];
h q[18];
rz(2.8489343524921047) q[18];
h q[18];
h q[19];
cx q[19],q[1];
rz(0.14994973362818875) q[1];
h q[1];
rz(2.8489343524921047) q[1];
h q[1];
cx q[19],q[1];
cx q[19],q[3];
rz(0.16385362163751527) q[3];
h q[3];
rz(2.8489343524921047) q[3];
h q[3];
cx q[19],q[3];
cx q[19],q[17];
rz(0.14576929355785317) q[17];
h q[17];
rz(2.8489343524921047) q[17];
h q[17];
cx q[19],q[17];
h q[19];
rz(2.8489343524921047) q[19];
h q[19];
h q[20];
cx q[20],q[8];
rz(0.15077557992522017) q[8];
h q[8];
rz(2.8489343524921047) q[8];
h q[8];
cx q[20],q[8];
cx q[20],q[12];
rz(0.15261570339056085) q[12];
h q[12];
rz(2.8489343524921047) q[12];
h q[12];
cx q[20],q[12];
h q[21];
cx q[21],q[5];
rz(0.15877132182339615) q[5];
h q[5];
rz(2.8489343524921047) q[5];
h q[5];
cx q[21],q[5];
cx q[21],q[20];
rz(0.16770254630169568) q[20];
h q[20];
rz(2.8489343524921047) q[20];
h q[20];
cx q[21],q[20];
h q[22];
cx q[22],q[2];
rz(0.1831228929959714) q[2];
h q[2];
rz(2.8489343524921047) q[2];
h q[2];
cx q[22],q[2];
cx q[13],q[22];
rz(0.17471253764395694) q[22];
cx q[13],q[22];
cx q[22],q[21];
rz(0.17608375539368026) q[21];
h q[21];
rz(2.8489343524921047) q[21];
h q[21];
cx q[22],q[21];
h q[22];
rz(2.8489343524921047) q[22];
h q[22];
h q[23];
cx q[23],q[4];
rz(0.178086350596117) q[4];
h q[4];
rz(2.8489343524921047) q[4];
h q[4];
cx q[23],q[4];
cx q[23],q[13];
rz(0.16438065910931776) q[13];
h q[13];
rz(2.8489343524921047) q[13];
h q[13];
cx q[23],q[13];
cx q[23],q[14];
rz(0.16509469921656184) q[14];
h q[14];
rz(2.8489343524921047) q[14];
h q[14];
cx q[23],q[14];
h q[23];
rz(2.8489343524921047) q[23];
h q[23];
