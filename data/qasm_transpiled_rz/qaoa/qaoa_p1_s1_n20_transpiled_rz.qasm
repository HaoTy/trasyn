OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
h q[0];
h q[1];
h q[2];
cx q[0],q[2];
rz(5.684931614831145) q[2];
cx q[0],q[2];
h q[3];
h q[4];
cx q[1],q[4];
rz(5.561952891438018) q[4];
cx q[1],q[4];
h q[5];
cx q[0],q[5];
rz(5.5883416436149895) q[5];
cx q[0],q[5];
cx q[2],q[5];
rz(5.454150790629129) q[5];
cx q[2],q[5];
h q[6];
h q[7];
cx q[7],q[2];
rz(0.6341976314840654) q[2];
h q[2];
rz(1.6307114266765907) q[2];
h q[2];
rz(3*pi) q[2];
cx q[7],q[2];
cx q[6],q[7];
rz(4.855072415156999) q[7];
cx q[6],q[7];
h q[8];
cx q[4],q[8];
rz(5.636602200628809) q[8];
cx q[4],q[8];
h q[9];
cx q[3],q[9];
rz(6.114401494286981) q[9];
cx q[3],q[9];
h q[10];
cx q[8],q[10];
rz(6.123750043040826) q[10];
cx q[8],q[10];
h q[11];
cx q[3],q[11];
rz(5.858173180001528) q[11];
cx q[3],q[11];
h q[12];
cx q[10],q[12];
rz(5.522173104763528) q[12];
cx q[10],q[12];
h q[13];
cx q[6],q[13];
rz(4.818970811789801) q[13];
cx q[6],q[13];
cx q[12],q[13];
rz(5.047863962591301) q[13];
cx q[12],q[13];
h q[14];
cx q[1],q[14];
rz(5.5539678889704955) q[14];
cx q[1],q[14];
cx q[14],q[4];
rz(-4.0729752487289455) q[4];
h q[4];
rz(1.6307114266765907) q[4];
h q[4];
rz(3*pi) q[4];
cx q[14],q[4];
cx q[11],q[14];
rz(-4.1935438407982595) q[14];
h q[14];
rz(1.6307114266765907) q[14];
h q[14];
rz(3*pi) q[14];
cx q[11],q[14];
h q[15];
cx q[9],q[15];
rz(5.412991119936015) q[15];
cx q[9],q[15];
cx q[15],q[11];
rz(-4.222440823694802) q[11];
h q[11];
rz(1.6307114266765907) q[11];
h q[11];
rz(3*pi) q[11];
cx q[15],q[11];
h q[16];
cx q[16],q[1];
rz(-3.863763381739576) q[1];
h q[1];
rz(1.6307114266765907) q[1];
h q[1];
rz(3*pi) q[1];
cx q[16],q[1];
cx q[16],q[7];
rz(-3.770780893786706) q[7];
h q[7];
rz(1.6307114266765907) q[7];
h q[7];
rz(3*pi) q[7];
cx q[16],q[7];
cx q[16],q[8];
rz(-4.224631958122785) q[8];
h q[8];
rz(1.6307114266765907) q[8];
h q[8];
rz(3*pi) q[8];
cx q[16],q[8];
h q[16];
rz(4.652473880502996) q[16];
h q[16];
h q[17];
cx q[17],q[5];
rz(-3.241765179293837) q[5];
h q[5];
rz(1.6307114266765907) q[5];
h q[5];
rz(3*pi) q[5];
cx q[17],q[5];
cx q[17],q[9];
rz(-4.601056683898065) q[9];
h q[9];
rz(1.6307114266765907) q[9];
h q[9];
rz(3*pi) q[9];
cx q[17],q[9];
cx q[17],q[15];
rz(-3.5672202738954355) q[15];
h q[15];
rz(1.6307114266765899) q[15];
h q[15];
rz(3*pi) q[15];
cx q[17],q[15];
h q[17];
rz(4.652473880502996) q[17];
h q[17];
h q[18];
cx q[18],q[0];
rz(1.367872365847619) q[0];
h q[0];
rz(1.6307114266765907) q[0];
h q[0];
rz(3*pi) q[0];
cx q[18],q[0];
cx q[18],q[3];
rz(-3.355557921081321) q[3];
h q[3];
rz(1.6307114266765899) q[3];
h q[3];
rz(3*pi) q[3];
cx q[18],q[3];
cx q[18],q[13];
rz(-4.483087812580084) q[13];
h q[13];
rz(1.6307114266765907) q[13];
h q[13];
rz(3*pi) q[13];
cx q[18],q[13];
h q[18];
rz(4.652473880502996) q[18];
h q[18];
h q[19];
cx q[19],q[6];
rz(-4.479944428950276) q[6];
h q[6];
rz(1.6307114266765907) q[6];
h q[6];
rz(3*pi) q[6];
cx q[19],q[6];
cx q[19],q[10];
rz(-3.1218173621729606) q[10];
h q[10];
rz(1.6307114266765907) q[10];
h q[10];
rz(3*pi) q[10];
cx q[19],q[10];
cx q[19],q[12];
rz(-2.7990200699651955) q[12];
h q[12];
rz(1.6307114266765907) q[12];
h q[12];
rz(3*pi) q[12];
cx q[19],q[12];
h q[19];
rz(4.652473880502996) q[19];
h q[19];
