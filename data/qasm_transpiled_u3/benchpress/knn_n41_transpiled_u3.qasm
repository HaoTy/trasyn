OPENQASM 2.0;
include "qelib1.inc";
qreg q0[41];
u3(pi/2,0,pi) q0[0];
u3(1.7044421049736478,0,0) q0[1];
u3(1.0958779629585713,0,0) q0[2];
u3(1.2309773709490481,0,0) q0[3];
u3(0.4876021266258924,0,0) q0[4];
u3(0.615096739850288,0,0) q0[5];
u3(1.8362803692692948,0,0) q0[6];
u3(0.17035702146686962,0,0) q0[7];
u3(0.0760181161010892,0,0) q0[8];
u3(0.9048445541342407,0,0) q0[9];
u3(0.35629343147681364,0,0) q0[10];
u3(1.1993952535302623,0,0) q0[11];
u3(0.2540130613766632,0,0) q0[12];
u3(0.47965848473659195,0,0) q0[13];
u3(0.12412549084739496,0,0) q0[14];
u3(0.8762899323374341,0,0) q0[15];
u3(2.076321021715302,0,0) q0[16];
u3(0.45901345447942404,0,0) q0[17];
u3(0.6772913953916866,0,0) q0[18];
u3(2.50357175527731,0,0) q0[19];
u3(0.9517094639363596,0,0) q0[20];
u3(0.8549329989283637,-pi/2,0) q0[21];
cx q0[21],q0[1];
u3(pi/2,0,-pi/2) q0[21];
cx q0[1],q0[21];
u3(0,0,-pi/4) q0[21];
cx q0[0],q0[21];
u3(0,pi/8,pi/8) q0[21];
cx q0[1],q0[21];
u3(0,0,pi/4) q0[1];
u3(0,-pi/8,-pi/8) q0[21];
cx q0[0],q0[21];
cx q0[0],q0[1];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[1];
cx q0[0],q0[1];
u3(pi/2,0,-3*pi/4) q0[21];
cx q0[21],q0[1];
u3(2.5377786539658453,-pi/2,0) q0[22];
cx q0[22],q0[2];
u3(pi/2,0,-pi/2) q0[22];
cx q0[2],q0[22];
u3(0,0,-pi/4) q0[22];
cx q0[0],q0[22];
u3(0,pi/8,pi/8) q0[22];
cx q0[2],q0[22];
u3(0,0,pi/4) q0[2];
u3(0,-pi/8,-pi/8) q0[22];
cx q0[0],q0[22];
cx q0[0],q0[2];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[2];
cx q0[0],q0[2];
u3(pi/2,0,-3*pi/4) q0[22];
cx q0[22],q0[2];
u3(1.7272597199680724,-pi/2,0) q0[23];
cx q0[23],q0[3];
u3(pi/2,0,-pi/2) q0[23];
cx q0[3],q0[23];
u3(0,0,-pi/4) q0[23];
cx q0[0],q0[23];
u3(0,pi/8,pi/8) q0[23];
cx q0[3],q0[23];
u3(0,0,pi/4) q0[3];
u3(0,-pi/8,-pi/8) q0[23];
cx q0[0],q0[23];
cx q0[0],q0[3];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[3];
cx q0[0],q0[3];
u3(pi/2,0,-3*pi/4) q0[23];
cx q0[23],q0[3];
u3(0.6329632976374331,-pi/2,0) q0[24];
cx q0[24],q0[4];
u3(pi/2,0,-pi/2) q0[24];
cx q0[4],q0[24];
u3(0,0,-pi/4) q0[24];
cx q0[0],q0[24];
u3(0,pi/8,pi/8) q0[24];
cx q0[4],q0[24];
u3(0,0,pi/4) q0[4];
u3(0,-pi/8,-pi/8) q0[24];
cx q0[0],q0[24];
cx q0[0],q0[4];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[4];
cx q0[0],q0[4];
u3(pi/2,0,-3*pi/4) q0[24];
cx q0[24],q0[4];
u3(3.1344638057987977,-pi/2,0) q0[25];
cx q0[25],q0[5];
u3(pi/2,0,-pi/2) q0[25];
cx q0[5],q0[25];
u3(0,0,-pi/4) q0[25];
cx q0[0],q0[25];
u3(0,pi/8,pi/8) q0[25];
cx q0[5],q0[25];
u3(0,0,pi/4) q0[5];
u3(0,-pi/8,-pi/8) q0[25];
cx q0[0],q0[25];
cx q0[0],q0[5];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[5];
cx q0[0],q0[5];
u3(pi/2,0,-3*pi/4) q0[25];
cx q0[25],q0[5];
u3(3.0412791505597867,-pi/2,0) q0[26];
cx q0[26],q0[6];
u3(pi/2,0,-pi/2) q0[26];
cx q0[6],q0[26];
u3(0,0,-pi/4) q0[26];
cx q0[0],q0[26];
u3(0,pi/8,pi/8) q0[26];
cx q0[6],q0[26];
u3(0,0,pi/4) q0[6];
u3(0,-pi/8,-pi/8) q0[26];
cx q0[0],q0[26];
cx q0[0],q0[6];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[6];
cx q0[0],q0[6];
u3(pi/2,0,-3*pi/4) q0[26];
cx q0[26],q0[6];
u3(2.5111584596518126,-pi/2,0) q0[27];
cx q0[27],q0[7];
u3(pi/2,0,-pi/2) q0[27];
cx q0[7],q0[27];
u3(0,0,-pi/4) q0[27];
cx q0[0],q0[27];
u3(0,pi/8,pi/8) q0[27];
cx q0[7],q0[27];
u3(0,0,pi/4) q0[7];
u3(0,-pi/8,-pi/8) q0[27];
cx q0[0],q0[27];
cx q0[0],q0[7];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[7];
cx q0[0],q0[7];
u3(pi/2,0,-3*pi/4) q0[27];
cx q0[27],q0[7];
u3(0.8477076457109352,-pi/2,0) q0[28];
cx q0[28],q0[8];
u3(pi/2,0,-pi/2) q0[28];
cx q0[8],q0[28];
u3(0,0,-pi/4) q0[28];
cx q0[0],q0[28];
u3(0,pi/8,pi/8) q0[28];
cx q0[8],q0[28];
u3(0,0,pi/4) q0[8];
u3(0,-pi/8,-pi/8) q0[28];
cx q0[0],q0[28];
cx q0[0],q0[8];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[8];
cx q0[0],q0[8];
u3(pi/2,0,-3*pi/4) q0[28];
cx q0[28],q0[8];
u3(0.7177241698504794,-pi/2,0) q0[29];
cx q0[29],q0[9];
u3(pi/2,0,-pi/2) q0[29];
cx q0[9],q0[29];
u3(0,0,-pi/4) q0[29];
cx q0[0],q0[29];
u3(0,pi/8,pi/8) q0[29];
cx q0[9],q0[29];
u3(0,0,pi/4) q0[9];
u3(0,-pi/8,-pi/8) q0[29];
cx q0[0],q0[29];
cx q0[0],q0[9];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[9];
cx q0[0],q0[9];
u3(pi/2,0,-3*pi/4) q0[29];
cx q0[29],q0[9];
u3(1.2665404563869145,-pi/2,0) q0[30];
cx q0[30],q0[10];
u3(pi/2,0,-pi/2) q0[30];
cx q0[10],q0[30];
u3(0,0,-pi/4) q0[30];
cx q0[0],q0[30];
u3(0,pi/8,pi/8) q0[30];
cx q0[10],q0[30];
u3(0,0,pi/4) q0[10];
u3(0,-pi/8,-pi/8) q0[30];
cx q0[0],q0[30];
cx q0[0],q0[10];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[10];
cx q0[0],q0[10];
u3(pi/2,0,-3*pi/4) q0[30];
cx q0[30],q0[10];
u3(1.3097989599488284,-pi/2,0) q0[31];
cx q0[31],q0[11];
u3(pi/2,0,-pi/2) q0[31];
cx q0[11],q0[31];
u3(0,0,-pi/4) q0[31];
cx q0[0],q0[31];
u3(0,pi/8,pi/8) q0[31];
cx q0[11],q0[31];
u3(0,0,pi/4) q0[11];
u3(0,-pi/8,-pi/8) q0[31];
cx q0[0],q0[31];
cx q0[0],q0[11];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[11];
cx q0[0],q0[11];
u3(pi/2,0,-3*pi/4) q0[31];
cx q0[31],q0[11];
u3(2.1652614066027747,-pi/2,0) q0[32];
cx q0[32],q0[12];
u3(pi/2,0,-pi/2) q0[32];
cx q0[12],q0[32];
u3(0,0,-pi/4) q0[32];
cx q0[0],q0[32];
u3(0,pi/8,pi/8) q0[32];
cx q0[12],q0[32];
u3(0,0,pi/4) q0[12];
u3(0,-pi/8,-pi/8) q0[32];
cx q0[0],q0[32];
cx q0[0],q0[12];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[12];
cx q0[0],q0[12];
u3(pi/2,0,-3*pi/4) q0[32];
cx q0[32],q0[12];
u3(0.07153878230580002,-pi/2,0) q0[33];
cx q0[33],q0[13];
u3(pi/2,0,-pi/2) q0[33];
cx q0[13],q0[33];
u3(0,0,-pi/4) q0[33];
cx q0[0],q0[33];
u3(0,pi/8,pi/8) q0[33];
cx q0[13],q0[33];
u3(0,0,pi/4) q0[13];
u3(0,-pi/8,-pi/8) q0[33];
cx q0[0],q0[33];
cx q0[0],q0[13];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[13];
cx q0[0],q0[13];
u3(pi/2,0,-3*pi/4) q0[33];
cx q0[33],q0[13];
u3(2.725902017084623,-pi/2,0) q0[34];
cx q0[34],q0[14];
u3(pi/2,0,-pi/2) q0[34];
cx q0[14],q0[34];
u3(0,0,-pi/4) q0[34];
cx q0[0],q0[34];
u3(0,pi/8,pi/8) q0[34];
cx q0[14],q0[34];
u3(0,0,pi/4) q0[14];
u3(0,-pi/8,-pi/8) q0[34];
cx q0[0],q0[34];
cx q0[0],q0[14];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[14];
cx q0[0],q0[14];
u3(pi/2,0,-3*pi/4) q0[34];
cx q0[34],q0[14];
u3(2.8993704288649926,-pi/2,0) q0[35];
cx q0[35],q0[15];
u3(pi/2,0,-pi/2) q0[35];
cx q0[15],q0[35];
u3(0,0,-pi/4) q0[35];
cx q0[0],q0[35];
u3(0,pi/8,pi/8) q0[35];
cx q0[15],q0[35];
u3(0,0,pi/4) q0[15];
u3(0,-pi/8,-pi/8) q0[35];
cx q0[0],q0[35];
cx q0[0],q0[15];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[15];
cx q0[0],q0[15];
u3(pi/2,0,-3*pi/4) q0[35];
cx q0[35],q0[15];
u3(2.543292869372673,-pi/2,0) q0[36];
cx q0[36],q0[16];
u3(pi/2,0,-pi/2) q0[36];
cx q0[16],q0[36];
u3(0,0,-pi/4) q0[36];
cx q0[0],q0[36];
u3(0,pi/8,pi/8) q0[36];
cx q0[16],q0[36];
u3(0,0,pi/4) q0[16];
u3(0,-pi/8,-pi/8) q0[36];
cx q0[0],q0[36];
cx q0[0],q0[16];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[16];
cx q0[0],q0[16];
u3(pi/2,0,-3*pi/4) q0[36];
cx q0[36],q0[16];
u3(1.0366165973566375,-pi/2,0) q0[37];
cx q0[37],q0[17];
u3(pi/2,0,-pi/2) q0[37];
cx q0[17],q0[37];
u3(0,0,-pi/4) q0[37];
cx q0[0],q0[37];
u3(0,pi/8,pi/8) q0[37];
cx q0[17],q0[37];
u3(0,0,pi/4) q0[17];
u3(0,-pi/8,-pi/8) q0[37];
cx q0[0],q0[37];
cx q0[0],q0[17];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[17];
cx q0[0],q0[17];
u3(pi/2,0,-3*pi/4) q0[37];
cx q0[37],q0[17];
u3(1.6750146490354798,-pi/2,0) q0[38];
cx q0[38],q0[18];
u3(pi/2,0,-pi/2) q0[38];
cx q0[18],q0[38];
u3(0,0,-pi/4) q0[38];
cx q0[0],q0[38];
u3(0,pi/8,pi/8) q0[38];
cx q0[18],q0[38];
u3(0,0,pi/4) q0[18];
u3(0,-pi/8,-pi/8) q0[38];
cx q0[0],q0[38];
cx q0[0],q0[18];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[18];
cx q0[0],q0[18];
u3(pi/2,0,-3*pi/4) q0[38];
cx q0[38],q0[18];
u3(0.8493287435327369,-pi/2,0) q0[39];
cx q0[39],q0[19];
u3(pi/2,0,-pi/2) q0[39];
cx q0[19],q0[39];
u3(0,0,-pi/4) q0[39];
cx q0[0],q0[39];
u3(0,pi/8,pi/8) q0[39];
cx q0[19],q0[39];
u3(0,0,pi/4) q0[19];
u3(0,-pi/8,-pi/8) q0[39];
cx q0[0],q0[39];
cx q0[0],q0[19];
u3(0,0,pi/4) q0[0];
u3(0,0,-pi/4) q0[19];
cx q0[0],q0[19];
u3(pi/2,0,-3*pi/4) q0[39];
cx q0[39],q0[19];
u3(1.4843570631924323,-pi/2,0) q0[40];
cx q0[40],q0[20];
u3(pi/2,0,-pi/2) q0[40];
cx q0[20],q0[40];
u3(0,0,-pi/4) q0[40];
cx q0[0],q0[40];
u3(0,pi/8,pi/8) q0[40];
cx q0[20],q0[40];
u3(0,0,pi/4) q0[20];
u3(0,-pi/8,-pi/8) q0[40];
cx q0[0],q0[40];
cx q0[0],q0[20];
u3(0,0,-pi/4) q0[20];
cx q0[0],q0[20];
u3(pi/2,0,-3*pi/4) q0[0];
u3(pi/2,0,-3*pi/4) q0[40];
cx q0[40],q0[20];
