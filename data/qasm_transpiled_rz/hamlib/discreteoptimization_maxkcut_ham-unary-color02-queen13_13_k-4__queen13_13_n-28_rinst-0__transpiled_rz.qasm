OPENQASM 2.0;
include "qelib1.inc";
qreg q[112];
rz(-0.75) q[0];
rz(-0.75) q[1];
rz(-0.75) q[2];
rz(-0.75) q[3];
rz(-1.25) q[4];
rz(-1.25) q[5];
rz(-1.25) q[6];
rz(-1.25) q[7];
rz(-0.75) q[8];
rz(-0.75) q[9];
rz(-0.75) q[10];
rz(-0.75) q[11];
rz(-1.25) q[12];
rz(-1.25) q[13];
rz(-1.25) q[14];
rz(-1.25) q[15];
rz(-1.25) q[16];
rz(-1.25) q[17];
rz(-1.25) q[18];
rz(-1.25) q[19];
cx q[15],q[19];
cx q[15],q[7];
rz(0.25) q[19];
cx q[3],q[15];
rz(0.25) q[7];
cx q[15],q[7];
cx q[15],q[19];
rz(0.25) q[19];
rz(0.25) q[7];
rz(-1.25) q[20];
rz(-1.25) q[21];
rz(-1.25) q[22];
rz(-1.25) q[23];
rz(-1.25) q[24];
rz(-1.25) q[25];
rz(-1.25) q[26];
rz(-1.25) q[27];
rz(-1.25) q[28];
rz(-1.25) q[29];
cx q[29],q[21];
cx q[17],q[29];
rz(0.25) q[21];
rz(0.25) q[29];
rz(-1.25) q[30];
rz(-1.25) q[31];
rz(-1.0) q[32];
rz(-1.0) q[33];
rz(-1.0) q[34];
rz(-1.0) q[35];
rz(-1.25) q[36];
rz(-1.25) q[37];
rz(-1.25) q[38];
rz(-1.25) q[39];
rz(-1.25) q[40];
rz(-1.25) q[41];
rz(-1.25) q[42];
rz(-1.25) q[43];
rz(-1.5) q[44];
cx q[44],q[40];
rz(0.25) q[40];
rz(-1.5) q[45];
cx q[41],q[45];
cx q[41],q[13];
rz(0.25) q[13];
cx q[41],q[5];
cx q[41],q[1];
rz(0.25) q[1];
rz(0.25) q[45];
cx q[41],q[45];
rz(0.25) q[5];
cx q[5],q[13];
rz(0.25) q[13];
cx q[5],q[1];
rz(0.25) q[1];
rz(-1.5) q[46];
cx q[42],q[46];
cx q[14],q[42];
cx q[14],q[6];
rz(0.25) q[42];
rz(0.25) q[46];
rz(0.25) q[6];
cx q[6],q[42];
rz(0.25) q[42];
cx q[6],q[2];
cx q[14],q[2];
rz(0.25) q[2];
cx q[42],q[2];
rz(0.25) q[2];
rz(-1.5) q[47];
cx q[47],q[43];
rz(0.25) q[43];
cx q[47],q[31];
rz(0.25) q[31];
rz(-1.25) q[48];
cx q[48],q[24];
rz(0.25) q[24];
cx q[48],q[8];
rz(0.25) q[8];
rz(-1.25) q[49];
rz(-1.25) q[50];
rz(-1.25) q[51];
rz(-1.75) q[52];
rz(-1.75) q[53];
rz(-1.75) q[54];
cx q[54],q[22];
rz(0.25) q[22];
cx q[54],q[22];
rz(-1.75) q[55];
rz(-1.75) q[56];
rz(-1.75) q[57];
rz(-1.75) q[58];
rz(-1.75) q[59];
rz(-1.5) q[60];
rz(-1.5) q[61];
rz(-1.5) q[62];
rz(-1.5) q[63];
rz(-2.25) q[64];
rz(-2.25) q[65];
rz(-2.25) q[66];
rz(-2.25) q[67];
rz(-1.0) q[68];
cx q[52],q[68];
cx q[52],q[56];
cx q[52],q[32];
cx q[20],q[52];
rz(0.25) q[32];
rz(0.25) q[52];
rz(0.25) q[56];
rz(0.25) q[68];
cx q[68],q[32];
rz(0.25) q[32];
rz(-1.0) q[69];
rz(-1.0) q[70];
rz(-1.0) q[71];
cx q[55],q[71];
rz(0.25) q[71];
rz(-7.783185307179586) q[72];
rz(-1.5) q[73];
rz(-1.5) q[74];
rz(-1.5) q[75];
cx q[67],q[75];
cx q[63],q[67];
cx q[63],q[55];
rz(0.25) q[55];
cx q[63],q[11];
rz(0.25) q[11];
rz(0.25) q[67];
rz(0.25) q[75];
cx q[67],q[75];
cx q[55],q[67];
cx q[55],q[35];
cx q[63],q[35];
rz(0.25) q[35];
rz(0.25) q[67];
cx q[71],q[35];
rz(0.25) q[35];
cx q[71],q[35];
cx q[55],q[35];
rz(0.25) q[75];
rz(-1.25) q[76];
rz(-1.25) q[77];
rz(-1.25) q[78];
rz(-1.25) q[79];
cx q[51],q[79];
cx q[59],q[51];
rz(0.25) q[51];
cx q[59],q[39];
cx q[27],q[59];
rz(0.25) q[39];
cx q[47],q[27];
rz(0.25) q[27];
cx q[47],q[23];
rz(0.25) q[23];
cx q[23],q[31];
rz(0.25) q[31];
cx q[47],q[3];
cx q[43],q[3];
rz(0.25) q[3];
cx q[3],q[15];
rz(0.25) q[15];
cx q[3],q[7];
cx q[3],q[19];
cx q[43],q[23];
cx q[23],q[15];
rz(0.25) q[15];
rz(0.25) q[59];
cx q[59],q[51];
rz(0.25) q[51];
cx q[59],q[39];
cx q[27],q[39];
rz(0.25) q[39];
rz(0.25) q[7];
cx q[23],q[7];
rz(0.25) q[7];
cx q[27],q[7];
cx q[27],q[59];
rz(0.25) q[79];
cx q[51],q[79];
rz(0.25) q[79];
rz(-1.0) q[80];
rz(-1.0) q[81];
rz(-1.0) q[82];
rz(-1.0) q[83];
rz(-1.25) q[84];
cx q[80],q[84];
cx q[80],q[64];
cx q[36],q[80];
cx q[44],q[36];
cx q[28],q[44];
cx q[28],q[20];
cx q[16],q[28];
cx q[16],q[12];
rz(0.25) q[12];
cx q[16],q[0];
rz(0.25) q[0];
cx q[16],q[4];
rz(0.25) q[20];
rz(0.25) q[28];
rz(0.25) q[36];
cx q[4],q[0];
rz(0.25) q[0];
cx q[4],q[12];
rz(0.25) q[12];
cx q[28],q[4];
rz(0.25) q[44];
cx q[20],q[44];
cx q[20],q[4];
rz(0.25) q[4];
cx q[4],q[12];
rz(0.25) q[12];
rz(0.25) q[44];
cx q[44],q[40];
cx q[4],q[40];
rz(0.25) q[40];
cx q[0],q[40];
rz(0.25) q[40];
cx q[0],q[40];
cx q[4],q[40];
cx q[40],q[12];
rz(0.25) q[12];
cx q[44],q[36];
rz(0.25) q[64];
rz(0.25) q[80];
rz(0.25) q[84];
cx q[80],q[84];
cx q[80],q[64];
rz(0.25) q[84];
cx q[20],q[84];
cx q[36],q[84];
cx q[36],q[56];
cx q[52],q[56];
rz(0.25) q[56];
cx q[56],q[64];
cx q[36],q[56];
rz(0.25) q[64];
cx q[52],q[64];
cx q[56],q[64];
rz(0.25) q[64];
rz(0.25) q[84];
cx q[56],q[84];
cx q[20],q[84];
cx q[28],q[20];
cx q[20],q[52];
cx q[52],q[68];
cx q[52],q[64];
rz(0.25) q[64];
rz(0.25) q[68];
rz(0.25) q[84];
rz(-1.25) q[85];
cx q[81],q[85];
rz(0.25) q[85];
rz(-1.25) q[86];
cx q[86],q[82];
rz(0.25) q[82];
cx q[86],q[58];
cx q[30],q[86];
cx q[30],q[18];
rz(0.25) q[18];
cx q[14],q[18];
cx q[30],q[18];
rz(0.25) q[18];
rz(0.25) q[58];
rz(0.25) q[86];
cx q[86],q[58];
rz(-1.25) q[87];
rz(-1.25) q[88];
rz(-1.25) q[89];
rz(-1.25) q[90];
rz(-1.25) q[91];
rz(-1.25) q[92];
cx q[92],q[88];
rz(0.25) q[88];
rz(-1.25) q[93];
cx q[89],q[93];
cx q[89],q[61];
cx q[37],q[89];
cx q[37],q[45];
rz(0.25) q[45];
rz(0.25) q[61];
rz(0.25) q[89];
rz(0.25) q[93];
rz(-1.25) q[94];
cx q[90],q[94];
cx q[90],q[38];
rz(0.25) q[38];
cx q[90],q[34];
rz(0.25) q[34];
cx q[54],q[34];
cx q[90],q[34];
rz(0.25) q[34];
rz(0.25) q[94];
rz(-1.25) q[95];
rz(-2.25) q[96];
rz(-2.25) q[97];
rz(-2.25) q[98];
rz(-2.25) q[99];
rz(-1.5) q[100];
rz(-1.5) q[101];
cx q[101],q[69];
cx q[65],q[101];
rz(0.25) q[101];
cx q[65],q[81];
cx q[57],q[65];
cx q[57],q[53];
cx q[49],q[57];
cx q[49],q[25];
rz(0.25) q[25];
cx q[49],q[9];
rz(0.25) q[53];
rz(0.25) q[57];
cx q[57],q[25];
rz(0.25) q[25];
cx q[49],q[57];
cx q[37],q[57];
cx q[37],q[33];
cx q[37],q[41];
rz(0.25) q[57];
rz(0.25) q[65];
cx q[65],q[53];
rz(0.25) q[53];
rz(0.25) q[69];
cx q[101],q[69];
cx q[69],q[53];
rz(0.25) q[53];
cx q[69],q[53];
rz(0.25) q[81];
cx q[65],q[81];
cx q[81],q[85];
cx q[57],q[81];
rz(0.25) q[81];
rz(0.25) q[85];
cx q[57],q[85];
cx q[37],q[57];
rz(0.25) q[85];
cx q[89],q[33];
rz(0.25) q[33];
cx q[89],q[93];
rz(0.25) q[9];
cx q[93],q[45];
cx q[41],q[93];
rz(0.25) q[45];
rz(0.25) q[93];
cx q[93],q[45];
rz(-1.5) q[102];
cx q[102],q[74];
cx q[102],q[70];
cx q[102],q[66];
cx q[102],q[62];
rz(0.25) q[62];
rz(0.25) q[66];
rz(0.25) q[70];
rz(0.25) q[74];
cx q[66],q[74];
cx q[62],q[66];
cx q[102],q[62];
rz(0.25) q[66];
rz(0.25) q[74];
cx q[66],q[74];
rz(0.25) q[74];
cx q[90],q[62];
rz(0.25) q[62];
cx q[90],q[62];
cx q[54],q[62];
cx q[54],q[70];
cx q[102],q[70];
rz(0.25) q[62];
cx q[62],q[66];
rz(0.25) q[66];
rz(0.25) q[70];
cx q[70],q[34];
rz(0.25) q[34];
rz(-1.5) q[103];
rz(-1.0) q[104];
rz(-1.0) q[105];
rz(-1.0) q[106];
rz(-1.0) q[107];
rz(-1.75) q[108];
cx q[104],q[108];
cx q[104],q[96];
cx q[104],q[76];
rz(0.25) q[108];
cx q[72],q[104];
cx q[72],q[100];
rz(0.25) q[100];
cx q[72],q[92];
cx q[72],q[60];
rz(0.25) q[60];
cx q[60],q[100];
rz(0.25) q[100];
cx q[72],q[48];
cx q[72],q[16];
cx q[16],q[64];
rz(0.25) q[64];
rz(0.25) q[76];
rz(0.25) q[92];
cx q[92],q[88];
cx q[60],q[88];
rz(0.25) q[88];
rz(0.25) q[96];
cx q[96],q[108];
rz(0.25) q[108];
cx q[76],q[96];
cx q[104],q[76];
rz(-4.46238898038469) q[104];
h q[104];
rz(pi/2) q[104];
h q[104];
rz(5*pi/2) q[104];
rz(0.25) q[96];
cx q[76],q[96];
cx q[48],q[76];
rz(0.25) q[76];
cx q[76],q[24];
rz(0.25) q[24];
cx q[76],q[24];
rz(0.25) q[96];
cx q[48],q[96];
cx q[60],q[48];
cx q[48],q[8];
rz(0.25) q[8];
cx q[28],q[8];
rz(0.25) q[96];
cx q[48],q[96];
cx q[100],q[96];
rz(0.25) q[96];
cx q[96],q[108];
rz(0.25) q[108];
cx q[100],q[108];
cx q[88],q[108];
rz(0.25) q[108];
cx q[88],q[108];
cx q[92],q[108];
cx q[60],q[108];
rz(0.25) q[108];
cx q[60],q[64];
cx q[16],q[60];
cx q[16],q[92];
h q[16];
rz(pi/2) q[16];
h q[16];
rz(5*pi/2) q[16];
cx q[20],q[92];
cx q[44],q[92];
cx q[52],q[60];
rz(0.25) q[60];
rz(0.25) q[64];
cx q[64],q[100];
rz(0.25) q[100];
cx q[100],q[96];
cx q[72],q[16];
rz(-3*pi/2) q[16];
h q[16];
rz(pi/2) q[16];
h q[16];
rz(5*pi/2) q[16];
rz(0.25) q[92];
cx q[44],q[92];
cx q[92],q[40];
rz(0.25) q[40];
cx q[40],q[12];
cx q[92],q[108];
rz(0.25) q[96];
cx q[64],q[96];
cx q[60],q[96];
cx q[60],q[8];
cx q[52],q[8];
cx q[64],q[100];
cx q[64],q[12];
cx q[68],q[100];
cx q[68],q[32];
h q[68];
rz(0.25) q[8];
cx q[92],q[12];
cx q[92],q[40];
h q[40];
h q[92];
rz(0.25) q[96];
cx q[20],q[96];
cx q[52],q[96];
cx q[52],q[60];
h q[52];
cx q[56],q[96];
rz(-4.46238898038469) q[96];
h q[96];
rz(pi/2) q[96];
h q[96];
rz(pi) q[96];
cx q[56],q[84];
cx q[48],q[56];
cx q[60],q[100];
rz(-4.46238898038469) q[100];
h q[100];
rz(pi/2) q[100];
h q[100];
rz(pi) q[100];
cx q[20],q[60];
cx q[20],q[32];
cx q[20],q[16];
rz(-pi) q[16];
cx q[72],q[16];
h q[72];
rz(pi/2) q[72];
h q[72];
rz(5*pi/2) q[72];
cx q[104],q[72];
rz(-3*pi/2) q[72];
h q[72];
rz(pi/2) q[72];
h q[72];
rz(3*pi/2) q[72];
h q[104];
rz(-pi) q[16];
cx q[84],q[108];
rz(0.25) q[108];
cx q[84],q[108];
cx q[80],q[108];
cx q[36],q[108];
rz(-4.46238898038469) q[108];
h q[108];
rz(pi/2) q[108];
h q[108];
rz(pi) q[108];
cx q[36],q[88];
cx q[60],q[88];
cx q[60],q[56];
rz(0.25) q[56];
cx q[56],q[24];
rz(0.25) q[24];
cx q[56],q[24];
cx q[48],q[24];
h q[56];
cx q[60],q[24];
cx q[4],q[24];
rz(0.25) q[24];
cx q[4],q[24];
cx q[16],q[4];
cx q[4],q[0];
cx q[44],q[24];
rz(0.25) q[24];
cx q[44],q[24];
h q[24];
h q[44];
cx q[60],q[12];
rz(0.25) q[12];
cx q[16],q[12];
cx q[20],q[16];
cx q[16],q[8];
h q[16];
rz(pi/2) q[16];
h q[16];
rz(5*pi/2) q[16];
h q[20];
cx q[28],q[8];
h q[28];
cx q[68],q[16];
cx q[100],q[68];
h q[100];
rz(pi/2) q[100];
h q[100];
rz(5*pi/2) q[100];
cx q[52],q[16];
cx q[28],q[16];
cx q[20],q[16];
rz(-3*pi/2) q[16];
h q[16];
rz(pi/2) q[16];
h q[16];
rz(3*pi/2) q[16];
h q[28];
h q[52];
h q[68];
h q[80];
h q[84];
rz(0.25) q[88];
cx q[36],q[88];
h q[36];
cx q[88],q[32];
rz(0.25) q[32];
cx q[88],q[32];
cx q[76],q[32];
cx q[48],q[32];
cx q[60],q[32];
rz(0.25) q[32];
cx q[60],q[48];
cx q[48],q[32];
h q[48];
cx q[60],q[12];
h q[60];
cx q[64],q[12];
h q[64];
cx q[76],q[32];
h q[32];
h q[76];
h q[88];
cx q[92],q[20];
cx q[88],q[20];
cx q[84],q[20];
cx q[60],q[20];
cx q[48],q[20];
cx q[44],q[20];
cx q[40],q[20];
cx q[36],q[20];
cx q[32],q[20];
cx q[24],q[20];
h q[20];
h q[24];
h q[32];
h q[40];
h q[44];
cx q[64],q[60];
h q[60];
h q[64];
cx q[76],q[48];
cx q[56],q[48];
h q[48];
h q[76];
cx q[80],q[36];
cx q[108],q[80];
h q[108];
rz(pi/2) q[108];
h q[108];
rz(5*pi/2) q[108];
h q[36];
h q[80];
h q[84];
h q[88];
h q[92];
cx q[96],q[56];
h q[96];
rz(pi/2) q[96];
h q[96];
rz(5*pi/2) q[96];
h q[56];
rz(-1.75) q[109];
cx q[105],q[109];
cx q[105],q[97];
cx q[105],q[77];
cx q[105],q[73];
rz(0.25) q[109];
rz(0.25) q[73];
rz(0.25) q[77];
rz(0.25) q[97];
cx q[97],q[109];
rz(0.25) q[109];
cx q[97],q[77];
rz(0.25) q[77];
cx q[97],q[73];
cx q[105],q[97];
cx q[49],q[97];
rz(0.25) q[73];
rz(0.25) q[97];
cx q[97],q[77];
cx q[49],q[97];
cx q[49],q[9];
cx q[57],q[97];
rz(0.25) q[77];
cx q[49],q[77];
rz(0.25) q[97];
cx q[97],q[65];
rz(0.25) q[65];
cx q[65],q[73];
cx q[65],q[101];
rz(0.25) q[101];
cx q[109],q[101];
rz(0.25) q[101];
cx q[65],q[53];
rz(0.25) q[53];
rz(0.25) q[73];
cx q[101],q[73];
cx q[109],q[73];
cx q[65],q[73];
rz(0.25) q[73];
cx q[97],q[65];
cx q[57],q[65];
cx q[17],q[65];
cx q[17],q[29];
cx q[29],q[9];
cx q[37],q[29];
cx q[29],q[85];
cx q[57],q[25];
cx q[57],q[97];
rz(0.25) q[65];
cx q[65],q[69];
rz(0.25) q[69];
cx q[17],q[69];
cx q[17],q[65];
cx q[77],q[25];
rz(0.25) q[25];
rz(0.25) q[85];
rz(0.25) q[9];
cx q[61],q[9];
cx q[61],q[101];
cx q[89],q[9];
cx q[29],q[9];
cx q[41],q[29];
cx q[29],q[45];
rz(0.25) q[45];
cx q[45],q[21];
rz(0.25) q[21];
cx q[29],q[45];
cx q[45],q[21];
cx q[5],q[21];
rz(6.533185307179586) q[21];
cx q[13],q[21];
cx q[89],q[33];
cx q[37],q[33];
cx q[77],q[33];
rz(0.25) q[33];
cx q[77],q[33];
cx q[33],q[69];
rz(-4.46238898038469) q[69];
h q[69];
rz(pi/2) q[69];
h q[69];
rz(pi) q[69];
cx q[33],q[53];
cx q[77],q[25];
rz(0.25) q[9];
cx q[97],q[53];
rz(0.25) q[53];
cx q[33],q[53];
h q[33];
cx q[69],q[33];
h q[69];
rz(pi/2) q[69];
h q[69];
rz(5*pi/2) q[69];
h q[33];
cx q[97],q[109];
cx q[37],q[109];
cx q[89],q[109];
rz(0.25) q[109];
cx q[89],q[81];
cx q[81],q[109];
rz(0.25) q[109];
cx q[81],q[109];
cx q[109],q[101];
cx q[101],q[73];
rz(-4.46238898038469) q[101];
h q[101];
rz(pi/2) q[101];
h q[101];
rz(5*pi/2) q[101];
rz(0.25) q[73];
cx q[93],q[109];
cx q[89],q[109];
cx q[41],q[109];
rz(0.25) q[109];
cx q[109],q[85];
cx q[29],q[85];
cx q[41],q[25];
cx q[37],q[25];
cx q[5],q[25];
cx q[25],q[45];
rz(-4.46238898038469) q[25];
h q[25];
rz(pi/2) q[25];
h q[25];
rz(5*pi/2) q[25];
cx q[5],q[45];
rz(-4.46238898038469) q[45];
h q[45];
rz(pi/2) q[45];
h q[45];
rz(pi) q[45];
cx q[89],q[61];
cx q[37],q[61];
cx q[37],q[41];
h q[37];
cx q[61],q[65];
rz(6.533185307179586) q[65];
cx q[61],q[53];
rz(0.25) q[53];
cx q[21],q[53];
rz(-4.46238898038469) q[21];
h q[21];
rz(pi/2) q[21];
h q[21];
rz(5*pi/2) q[21];
cx q[89],q[81];
h q[81];
h q[89];
cx q[89],q[37];
cx q[81],q[37];
h q[37];
h q[81];
h q[89];
cx q[93],q[85];
rz(0.25) q[85];
cx q[93],q[73];
cx q[61],q[73];
cx q[41],q[73];
cx q[41],q[5];
cx q[17],q[5];
cx q[41],q[29];
h q[41];
cx q[5],q[1];
rz(0.25) q[1];
cx q[5],q[13];
rz(0.25) q[13];
cx q[61],q[13];
cx q[17],q[13];
cx q[13],q[65];
rz(-4.46238898038469) q[65];
h q[65];
rz(pi/2) q[65];
h q[65];
rz(5*pi/2) q[65];
cx q[13],q[53];
rz(-4.46238898038469) q[53];
h q[53];
rz(pi/2) q[53];
h q[53];
rz(pi) q[53];
cx q[17],q[5];
cx q[17],q[1];
h q[5];
cx q[25],q[5];
cx q[45],q[25];
h q[45];
rz(pi/2) q[45];
h q[45];
rz(5*pi/2) q[45];
h q[25];
h q[5];
cx q[61],q[13];
h q[13];
cx q[61],q[9];
h q[61];
cx q[101],q[61];
h q[101];
h q[61];
cx q[65],q[13];
cx q[21],q[13];
h q[13];
cx q[53],q[21];
h q[53];
rz(pi/2) q[53];
h q[53];
rz(5*pi/2) q[53];
h q[21];
h q[65];
rz(0.25) q[73];
cx q[93],q[85];
cx q[109],q[85];
h q[109];
h q[85];
cx q[93],q[73];
h q[73];
h q[93];
cx q[93],q[41];
cx q[109],q[93];
h q[109];
cx q[85],q[41];
cx q[73],q[41];
h q[41];
h q[73];
h q[85];
h q[93];
rz(-1.75) q[110];
cx q[106],q[110];
cx q[106],q[98];
rz(0.25) q[110];
cx q[78],q[106];
rz(0.25) q[106];
cx q[50],q[78];
cx q[50],q[26];
rz(0.25) q[26];
cx q[50],q[10];
rz(0.25) q[10];
cx q[50],q[10];
cx q[30],q[10];
rz(0.25) q[10];
cx q[10],q[62];
cx q[30],q[22];
rz(0.25) q[22];
cx q[14],q[22];
cx q[30],q[22];
rz(0.25) q[22];
cx q[6],q[22];
rz(0.25) q[22];
cx q[22],q[46];
cx q[42],q[46];
rz(0.25) q[46];
cx q[22],q[46];
h q[22];
rz(0.25) q[78];
cx q[78],q[26];
rz(0.25) q[26];
rz(0.25) q[98];
cx q[110],q[98];
cx q[106],q[110];
rz(0.25) q[98];
cx q[110],q[98];
rz(0.25) q[98];
cx q[78],q[98];
cx q[78],q[110];
rz(0.25) q[98];
cx q[50],q[98];
cx q[102],q[98];
cx q[50],q[110];
rz(0.25) q[98];
cx q[102],q[98];
cx q[102],q[110];
rz(0.25) q[110];
cx q[102],q[110];
cx q[54],q[98];
cx q[54],q[14];
cx q[30],q[54];
cx q[30],q[38];
cx q[54],q[62];
cx q[54],q[58];
rz(0.25) q[58];
cx q[62],q[74];
rz(-4.46238898038469) q[62];
h q[62];
rz(pi/2) q[62];
h q[62];
rz(5*pi/2) q[62];
cx q[10],q[74];
cx q[86],q[38];
cx q[90],q[38];
rz(0.25) q[38];
cx q[82],q[38];
rz(0.25) q[38];
cx q[90],q[110];
rz(0.25) q[110];
cx q[110],q[94];
cx q[90],q[110];
cx q[30],q[110];
cx q[30],q[50];
cx q[86],q[110];
rz(6.533185307179586) q[110];
cx q[82],q[110];
cx q[86],q[82];
h q[86];
rz(0.25) q[94];
cx q[110],q[94];
rz(-4.46238898038469) q[110];
h q[110];
rz(pi/2) q[110];
h q[110];
rz(5*pi/2) q[110];
rz(0.25) q[98];
cx q[66],q[98];
cx q[14],q[66];
cx q[14],q[70];
rz(0.25) q[66];
cx q[66],q[18];
rz(0.25) q[18];
cx q[66],q[18];
cx q[14],q[66];
cx q[66],q[58];
rz(0.25) q[58];
cx q[70],q[18];
rz(0.25) q[18];
cx q[2],q[18];
cx q[42],q[18];
cx q[70],q[18];
cx q[6],q[18];
rz(-4.46238898038469) q[18];
h q[18];
rz(pi/2) q[18];
h q[18];
rz(pi) q[18];
cx q[6],q[46];
cx q[70],q[34];
h q[70];
cx q[82],q[66];
cx q[54],q[66];
cx q[54],q[14];
cx q[14],q[46];
rz(0.25) q[46];
h q[54];
rz(0.25) q[66];
rz(0.25) q[98];
cx q[98],q[58];
rz(0.25) q[58];
cx q[66],q[98];
cx q[98],q[58];
cx q[38],q[58];
rz(0.25) q[58];
cx q[38],q[58];
cx q[50],q[58];
cx q[82],q[58];
rz(0.25) q[58];
cx q[26],q[58];
cx q[78],q[58];
rz(-4.46238898038469) q[58];
h q[58];
rz(pi/2) q[58];
h q[58];
rz(pi) q[58];
cx q[50],q[78];
h q[50];
cx q[78],q[14];
cx q[14],q[34];
cx q[14],q[6];
rz(0.25) q[34];
cx q[6],q[26];
cx q[82],q[46];
cx q[38],q[46];
rz(0.25) q[46];
cx q[38],q[46];
cx q[82],q[74];
cx q[94],q[46];
rz(0.25) q[46];
cx q[98],q[74];
rz(0.25) q[74];
cx q[98],q[74];
cx q[94],q[74];
rz(0.25) q[74];
cx q[106],q[74];
cx q[82],q[94];
cx q[78],q[94];
cx q[78],q[30];
cx q[82],q[66];
h q[66];
cx q[82],q[38];
h q[38];
h q[82];
cx q[94],q[74];
cx q[6],q[94];
cx q[30],q[6];
cx q[30],q[14];
h q[14];
cx q[42],q[94];
cx q[6],q[2];
h q[6];
cx q[70],q[14];
h q[14];
h q[70];
rz(0.25) q[74];
cx q[78],q[30];
cx q[30],q[10];
h q[10];
h q[30];
cx q[62],q[10];
h q[10];
h q[62];
cx q[78],q[74];
cx q[106],q[74];
h q[106];
h q[74];
cx q[78],q[34];
h q[34];
h q[78];
cx q[86],q[30];
cx q[82],q[30];
cx q[110],q[82];
h q[110];
cx q[78],q[30];
cx q[106],q[78];
h q[106];
cx q[74],q[30];
cx q[66],q[30];
cx q[54],q[30];
cx q[50],q[30];
cx q[38],q[30];
cx q[34],q[30];
h q[30];
h q[34];
h q[38];
h q[50];
h q[54];
h q[66];
h q[74];
h q[78];
h q[86];
cx q[94],q[46];
rz(-4.46238898038469) q[94];
h q[94];
rz(pi/2) q[94];
h q[94];
rz(5*pi/2) q[94];
cx q[26],q[46];
rz(-4.46238898038469) q[26];
h q[26];
rz(pi/2) q[26];
h q[26];
rz(5*pi/2) q[26];
cx q[42],q[46];
rz(-4.46238898038469) q[46];
h q[46];
rz(pi/2) q[46];
h q[46];
rz(pi) q[46];
cx q[42],q[2];
h q[2];
cx q[18],q[2];
h q[18];
rz(pi/2) q[18];
h q[18];
rz(5*pi/2) q[18];
h q[2];
h q[42];
cx q[42],q[6];
cx q[26],q[6];
cx q[22],q[6];
h q[22];
cx q[58],q[26];
h q[58];
rz(pi/2) q[58];
h q[58];
rz(5*pi/2) q[58];
cx q[46],q[26];
h q[46];
rz(pi/2) q[46];
h q[46];
rz(5*pi/2) q[46];
h q[26];
h q[6];
cx q[94],q[42];
h q[42];
h q[94];
h q[98];
cx q[98],q[82];
h q[82];
h q[98];
rz(-1.75) q[111];
cx q[111],q[107];
rz(0.25) q[107];
cx q[111],q[103];
rz(0.25) q[103];
cx q[111],q[99];
cx q[111],q[95];
cx q[111],q[91];
cx q[111],q[87];
cx q[111],q[83];
rz(0.25) q[83];
rz(0.25) q[87];
cx q[87],q[83];
rz(0.25) q[83];
rz(0.25) q[91];
rz(0.25) q[95];
cx q[91],q[95];
cx q[111],q[91];
cx q[63],q[91];
rz(0.25) q[91];
rz(0.25) q[95];
cx q[91],q[95];
cx q[35],q[91];
cx q[75],q[95];
cx q[63],q[75];
rz(0.25) q[91];
rz(0.25) q[95];
cx q[47],q[95];
cx q[75],q[95];
rz(6.533185307179586) q[95];
cx q[43],q[95];
rz(-4.46238898038469) q[95];
h q[95];
rz(pi/2) q[95];
h q[95];
rz(5*pi/2) q[95];
cx q[43],q[7];
cx q[23],q[7];
cx q[23],q[31];
cx q[31],q[19];
rz(0.25) q[19];
cx q[43],q[31];
cx q[43],q[23];
rz(0.25) q[7];
rz(0.25) q[99];
cx q[99],q[103];
rz(0.25) q[103];
cx q[107],q[99];
cx q[107],q[75];
cx q[111],q[75];
cx q[111],q[87];
cx q[47],q[87];
cx q[47],q[63];
rz(0.25) q[75];
cx q[87],q[39];
rz(0.25) q[39];
cx q[83],q[39];
rz(0.25) q[39];
cx q[87],q[59];
rz(0.25) q[59];
cx q[87],q[31];
rz(0.25) q[31];
cx q[11],q[31];
cx q[11],q[51];
cx q[63],q[87];
cx q[87],q[31];
cx q[31],q[19];
rz(-4.46238898038469) q[31];
h q[31];
rz(pi/2) q[31];
h q[31];
rz(5*pi/2) q[31];
cx q[11],q[19];
cx q[87],q[59];
cx q[55],q[59];
cx q[55],q[23];
rz(0.25) q[59];
cx q[63],q[23];
rz(0.25) q[23];
cx q[23],q[15];
cx q[63],q[27];
cx q[27],q[51];
cx q[27],q[79];
cx q[35],q[79];
rz(0.25) q[51];
cx q[63],q[107];
cx q[47],q[107];
cx q[111],q[107];
cx q[63],q[47];
cx q[47],q[27];
cx q[27],q[7];
cx q[47],q[43];
cx q[63],q[43];
cx q[43],q[3];
h q[43];
cx q[67],q[59];
rz(0.25) q[59];
cx q[67],q[15];
rz(0.25) q[15];
cx q[55],q[67];
rz(0.25) q[79];
cx q[87],q[83];
cx q[83],q[67];
rz(0.25) q[67];
h q[87];
cx q[91],q[39];
cx q[35],q[39];
cx q[35],q[79];
cx q[107],q[79];
cx q[47],q[35];
h q[35];
rz(0.25) q[79];
cx q[83],q[39];
rz(0.25) q[39];
cx q[83],q[67];
cx q[67],q[19];
rz(0.25) q[19];
cx q[71],q[19];
cx q[67],q[19];
cx q[55],q[19];
rz(0.25) q[19];
h q[83];
cx q[91],q[39];
h q[39];
h q[91];
cx q[91],q[35];
cx q[39],q[35];
h q[35];
h q[39];
h q[91];
cx q[95],q[43];
h q[43];
h q[95];
rz(0.25) q[99];
cx q[75],q[99];
rz(0.25) q[99];
cx q[99],q[103];
rz(0.25) q[103];
cx q[107],q[103];
cx q[75],q[103];
rz(0.25) q[103];
cx q[67],q[103];
rz(0.25) q[103];
cx q[71],q[103];
cx q[75],q[99];
cx q[99],q[79];
cx q[107],q[99];
cx q[107],q[75];
h q[107];
cx q[55],q[99];
h q[75];
rz(0.25) q[79];
rz(0.25) q[99];
cx q[55],q[99];
cx q[55],q[103];
cx q[55],q[47];
cx q[47],q[23];
cx q[47],q[19];
cx q[55],q[47];
cx q[47],q[15];
cx q[67],q[99];
cx q[67],q[103];
rz(-4.46238898038469) q[103];
h q[103];
rz(pi/2) q[103];
h q[103];
rz(pi) q[103];
cx q[71],q[19];
h q[71];
rz(0.25) q[99];
cx q[59],q[99];
rz(0.25) q[99];
cx q[99],q[51];
cx q[59],q[51];
cx q[11],q[51];
cx q[47],q[11];
h q[11];
cx q[31],q[11];
h q[11];
h q[31];
cx q[63],q[59];
cx q[63],q[55];
h q[55];
cx q[63],q[47];
h q[47];
h q[63];
cx q[63],q[47];
cx q[107],q[63];
h q[107];
cx q[67],q[51];
rz(0.25) q[51];
cx q[67],q[59];
cx q[59],q[51];
h q[59];
cx q[59],q[47];
cx q[55],q[47];
cx q[67],q[15];
h q[67];
cx q[71],q[55];
cx q[103],q[71];
h q[103];
rz(pi/2) q[103];
h q[103];
rz(5*pi/2) q[103];
h q[55];
h q[71];
cx q[87],q[63];
cx q[83],q[63];
cx q[75],q[63];
cx q[67],q[63];
h q[63];
h q[67];
h q[75];
h q[83];
h q[87];
cx q[99],q[79];
h q[79];
cx q[99],q[51];
h q[51];
cx q[51],q[47];
h q[47];
h q[51];
h q[99];
cx q[99],q[59];
cx q[79],q[59];
h q[59];
h q[79];
h q[99];
