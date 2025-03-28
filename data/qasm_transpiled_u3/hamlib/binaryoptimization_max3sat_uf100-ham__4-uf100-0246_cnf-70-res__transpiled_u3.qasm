OPENQASM 2.0;
include "qelib1.inc";
qreg q[70];
u3(0,0,-0.875) q[0];
u3(0,0,-0.25) q[1];
u3(0,0,-0.375) q[2];
u3(0,0,-0.25) q[3];
u3(0,0,0.75) q[4];
u3(0,0,0.75) q[5];
u3(0,0,-1.25) q[6];
u3(0,0,0.25) q[7];
u3(0,0,0.5) q[8];
u3(0,0,0.5) q[10];
u3(0,0,0.25) q[12];
u3(0,0,-0.875) q[13];
u3(0,0,-0.75) q[15];
u3(0,0,-0.25) q[16];
u3(0,0,-0.25) q[17];
u3(0,0,-0.25) q[18];
cx q[18],q[14];
u3(0,0,-0.125) q[14];
u3(0,0,0.375) q[19];
u3(0,0,-0.375) q[20];
u3(0,0,0.25) q[21];
u3(0,0,-0.125) q[22];
u3(0,0,-0.5) q[23];
u3(0,0,-0.5) q[24];
u3(0,0,0.375) q[25];
u3(0,0,-0.25) q[26];
u3(0,0,-0.625) q[27];
cx q[27],q[16];
u3(0,0,-0.125) q[16];
u3(0,0,0.375) q[28];
u3(0,0,1.0) q[29];
u3(0,0,0.5) q[30];
u3(0,0,-0.5) q[31];
u3(0,0,-0.375) q[32];
u3(0,0,-0.25) q[33];
u3(0,0,0.5) q[34];
u3(0,0,-0.125) q[35];
u3(0,0,-0.5) q[36];
u3(0,0,1.0) q[37];
u3(0,0,-0.5) q[38];
u3(0,0,0.125) q[39];
u3(0,0,-0.25) q[40];
cx q[40],q[34];
u3(0,0,0.125) q[34];
cx q[40],q[28];
u3(0,0,0.125) q[28];
cx q[40],q[22];
u3(0,0,-0.125) q[22];
cx q[28],q[22];
u3(0,0,-0.125) q[22];
cx q[40],q[20];
u3(0,0,-0.125) q[20];
cx q[40],q[22];
u3(0,0,0.125) q[22];
cx q[40],q[31];
cx q[31],q[28];
u3(0,0,0.125) q[28];
cx q[34],q[31];
u3(0,0,0.25) q[31];
u3(0,0,-0.5) q[41];
u3(0,0,0.5) q[42];
u3(0,0,-0.75) q[43];
u3(0,0,0.5) q[44];
u3(0,0,-0.75) q[45];
u3(0,0,0.125) q[46];
u3(0,0,0.875) q[47];
u3(0,0,-1.0) q[48];
u3(0,0,-0.125) q[49];
u3(0,0,1.125) q[51];
u3(0,0,0.625) q[52];
u3(0,0,0.25) q[53];
u3(0,0,-0.25) q[54];
u3(0,0,0.375) q[55];
u3(0,0,1.125) q[56];
u3(0,0,-1.125) q[57];
cx q[57],q[50];
u3(0,0,0.125) q[50];
cx q[57],q[35];
u3(0,0,0.125) q[35];
cx q[57],q[33];
u3(0,0,0.125) q[33];
cx q[57],q[25];
u3(0,0,-0.125) q[25];
cx q[35],q[25];
u3(0,0,-0.25) q[25];
cx q[57],q[17];
u3(0,0,0.125) q[17];
cx q[35],q[17];
u3(0,0,-0.125) q[17];
cx q[57],q[7];
u3(0,0,-0.125) q[7];
cx q[35],q[7];
u3(0,0,-0.25) q[7];
cx q[50],q[7];
cx q[35],q[7];
u3(0,0,0.125) q[7];
cx q[35],q[10];
cx q[57],q[17];
u3(0,0,0.125) q[17];
cx q[35],q[17];
cx q[28],q[17];
u3(0,0,0.125) q[17];
cx q[28],q[17];
u3(0,0,0.875) q[59];
u3(0,0,-0.125) q[60];
u3(0,0,0.125) q[61];
u3(0,0,1.0) q[62];
u3(0,0,0.125) q[63];
cx q[63],q[60];
u3(0,0,-0.125) q[60];
cx q[63],q[48];
u3(0,0,0.25) q[48];
cx q[63],q[23];
u3(0,0,0.125) q[23];
cx q[63],q[4];
u3(0,0,-0.125) q[4];
cx q[23],q[4];
u3(0,0,-0.125) q[4];
cx q[48],q[4];
cx q[23],q[4];
u3(0,0,-0.125) q[4];
cx q[63],q[23];
u3(0,0,0.625) q[64];
cx q[64],q[51];
u3(0,0,0.125) q[51];
cx q[51],q[40];
cx q[64],q[41];
u3(0,0,0.25) q[41];
cx q[64],q[5];
u3(0,0,0.125) q[5];
u3(0,0,-1.5) q[65];
u3(0,0,-0.125) q[66];
cx q[66],q[65];
u3(0,0,-0.25) q[65];
cx q[66],q[58];
u3(0,0,0.125) q[58];
cx q[66],q[54];
u3(0,0,0.125) q[54];
cx q[66],q[53];
u3(0,0,-0.125) q[53];
cx q[66],q[52];
u3(0,0,0.25) q[52];
cx q[53],q[52];
u3(0,0,-0.125) q[52];
cx q[66],q[47];
u3(0,0,0.125) q[47];
cx q[66],q[21];
u3(0,0,-0.125) q[21];
cx q[47],q[21];
u3(0,0,-0.125) q[21];
cx q[66],q[6];
u3(0,0,-0.125) q[6];
cx q[47],q[6];
u3(0,0,0.125) q[6];
cx q[66],q[1];
u3(0,0,-0.125) q[1];
cx q[47],q[1];
u3(0,0,-0.125) q[1];
cx q[21],q[1];
u3(0,0,-0.125) q[1];
cx q[47],q[21];
cx q[66],q[0];
u3(0,0,-0.25) q[0];
cx q[54],q[0];
u3(0,0,-0.125) q[0];
cx q[54],q[21];
u3(0,0,0.125) q[21];
cx q[54],q[21];
cx q[65],q[21];
u3(0,0,0.125) q[21];
cx q[66],q[6];
u3(0,0,0.125) q[6];
cx q[66],q[1];
u3(0,0,0.125) q[1];
u3(0,0,0.375) q[67];
u3(0,0,0.25) q[68];
cx q[68],q[67];
u3(0,0,0.375) q[67];
cx q[68],q[62];
u3(0,0,-0.25) q[62];
cx q[68],q[56];
u3(0,0,0.125) q[56];
cx q[68],q[44];
u3(0,0,-0.125) q[44];
cx q[56],q[44];
u3(0,0,0.125) q[44];
cx q[68],q[43];
u3(0,0,-0.25) q[43];
cx q[62],q[43];
u3(0,0,-0.125) q[43];
cx q[68],q[42];
u3(0,0,-0.25) q[42];
cx q[67],q[42];
u3(0,0,-0.125) q[42];
cx q[42],q[26];
u3(0,0,-0.125) q[26];
cx q[68],q[39];
u3(0,0,-0.25) q[39];
cx q[56],q[39];
u3(0,0,-0.25) q[39];
cx q[44],q[39];
u3(0,0,0.125) q[39];
cx q[68],q[37];
u3(0,0,0.125) q[37];
cx q[62],q[37];
u3(0,0,-0.25) q[37];
cx q[68],q[36];
u3(0,0,0.125) q[36];
cx q[62],q[36];
u3(0,0,-0.25) q[36];
cx q[68],q[32];
u3(0,0,0.125) q[32];
cx q[67],q[32];
u3(0,0,-0.125) q[32];
cx q[68],q[30];
u3(0,0,-0.125) q[30];
cx q[56],q[30];
u3(0,0,-0.125) q[30];
cx q[68],q[29];
u3(0,0,-0.125) q[29];
cx q[56],q[29];
u3(0,0,0.25) q[29];
cx q[68],q[13];
u3(0,0,0.125) q[13];
cx q[67],q[13];
u3(0,0,0.125) q[13];
cx q[55],q[13];
u3(0,0,-0.125) q[13];
cx q[68],q[9];
u3(0,0,0.125) q[9];
cx q[67],q[9];
u3(0,0,0.25) q[9];
cx q[42],q[9];
u3(0,0,0.25) q[9];
cx q[68],q[3];
u3(0,0,-0.25) q[3];
cx q[56],q[3];
u3(0,0,0.125) q[3];
cx q[68],q[2];
u3(0,0,-0.125) q[2];
cx q[62],q[2];
u3(0,0,0.25) q[2];
cx q[67],q[2];
cx q[62],q[2];
u3(0,0,0.125) q[2];
cx q[68],q[3];
u3(0,0,-0.125) q[3];
cx q[68],q[2];
u3(0,0,-0.125) q[2];
cx q[68],q[67];
cx q[68],q[39];
cx q[56],q[39];
u3(0,0,0.125) q[39];
cx q[44],q[39];
cx q[9],q[3];
cx q[56],q[3];
u3(0,0,0.125) q[3];
cx q[69],q[61];
u3(0,0,0.125) q[61];
cx q[69],q[59];
u3(0,0,-0.125) q[59];
cx q[69],q[49];
u3(0,0,0.125) q[49];
cx q[69],q[46];
u3(0,0,0.125) q[46];
cx q[49],q[46];
u3(0,0,-0.25) q[46];
cx q[69],q[45];
u3(0,0,-0.125) q[45];
cx q[49],q[45];
u3(0,0,-0.125) q[45];
cx q[69],q[38];
u3(0,0,0.125) q[38];
cx q[49],q[38];
u3(0,0,-0.125) q[38];
cx q[45],q[38];
u3(0,0,-0.25) q[38];
cx q[45],q[38];
cx q[69],q[24];
u3(0,0,0.125) q[24];
cx q[59],q[24];
u3(0,0,0.125) q[24];
cx q[69],q[19];
u3(0,0,-0.125) q[19];
cx q[61],q[19];
u3(0,0,-0.125) q[19];
cx q[64],q[19];
u3(0,0,0.125) q[19];
cx q[64],q[5];
cx q[64],q[40];
u3(0,0,0.125) q[40];
cx q[69],q[15];
u3(0,0,0.125) q[15];
cx q[33],q[15];
cx q[57],q[15];
u3(0,0,0.125) q[15];
cx q[69],q[12];
u3(0,0,-0.125) q[12];
cx q[49],q[12];
u3(0,0,-0.125) q[12];
cx q[49],q[12];
cx q[49],q[46];
cx q[46],q[18];
u3(0,0,0.125) q[18];
cx q[59],q[12];
u3(0,0,-0.125) q[12];
cx q[69],q[11];
u3(0,0,-0.125) q[11];
cx q[69],q[8];
u3(0,0,-0.125) q[8];
cx q[11],q[8];
u3(0,0,-0.25) q[8];
cx q[61],q[8];
cx q[11],q[8];
u3(0,0,-0.125) q[8];
cx q[69],q[38];
u3(0,0,0.125) q[38];
cx q[69],q[8];
u3(0,0,0.125) q[8];
cx q[69],q[12];
u3(0,0,0.125) q[12];
cx q[69],q[18];
u3(0,0,-0.125) q[18];
cx q[69],q[15];
u3(0,0,0.125) q[15];
cx q[33],q[15];
u3(0,0,0.125) q[15];
cx q[50],q[15];
u3(0,0,0.125) q[15];
cx q[57],q[15];
u3(0,0,0.125) q[15];
cx q[57],q[10];
u3(0,0,-0.125) q[10];
cx q[25],q[10];
u3(0,0,0.125) q[10];
cx q[57],q[23];
u3(0,0,0.125) q[23];
cx q[55],q[23];
cx q[55],q[47];
cx q[57],q[23];
u3(0,0,0.125) q[23];
cx q[60],q[57];
cx q[63],q[57];
u3(0,0,-0.125) q[57];
cx q[57],q[33];
u3(0,0,-0.25) q[33];
cx q[63],q[5];
u3(0,0,0.125) q[5];
cx q[60],q[5];
u3(0,0,0.125) q[5];
cx q[66],q[47];
u3(0,0,-0.125) q[47];
cx q[66],q[53];
cx q[64],q[53];
u3(0,0,-0.125) q[53];
cx q[64],q[53];
cx q[63],q[53];
u3(0,0,-0.25) q[53];
cx q[53],q[52];
u3(0,0,-0.125) q[52];
cx q[62],q[52];
cx q[63],q[53];
cx q[53],q[27];
u3(0,0,0.125) q[27];
cx q[66],q[65];
cx q[65],q[63];
u3(0,0,-0.125) q[63];
cx q[65],q[60];
u3(0,0,0.125) q[60];
cx q[63],q[60];
cx q[65],q[57];
u3(0,0,0.125) q[57];
cx q[65],q[60];
u3(0,0,0.25) q[60];
cx q[60],q[21];
u3(0,0,0.125) q[21];
cx q[60],q[33];
u3(0,0,-0.125) q[33];
cx q[65],q[60];
cx q[60],q[57];
u3(0,0,0.125) q[57];
cx q[57],q[50];
u3(0,0,0.125) q[50];
cx q[65],q[63];
cx q[67],q[55];
u3(0,0,0.125) q[55];
cx q[55],q[47];
u3(0,0,0.125) q[47];
cx q[47],q[32];
u3(0,0,0.125) q[32];
cx q[67],q[13];
u3(0,0,-0.125) q[13];
cx q[23],q[13];
u3(0,0,0.125) q[13];
cx q[67],q[10];
u3(0,0,0.125) q[10];
cx q[67],q[32];
u3(0,0,-0.125) q[32];
cx q[67],q[63];
u3(0,0,0.125) q[63];
cx q[67],q[63];
cx q[63],q[13];
u3(0,0,-0.125) q[13];
cx q[67],q[42];
cx q[42],q[26];
u3(0,0,0.125) q[26];
cx q[68],q[52];
u3(0,0,0.125) q[52];
cx q[63],q[52];
u3(0,0,-0.125) q[52];
cx q[63],q[62];
cx q[68],q[56];
cx q[56],q[21];
u3(0,0,-0.125) q[21];
cx q[60],q[56];
u3(0,0,0.125) q[56];
cx q[60],q[21];
u3(0,0,-0.125) q[21];
cx q[29],q[21];
u3(0,0,-0.125) q[21];
cx q[68],q[62];
u3(0,0,-0.125) q[62];
cx q[62],q[43];
u3(0,0,0.125) q[43];
cx q[63],q[43];
cx q[69],q[66];
cx q[66],q[11];
u3(0,0,-0.125) q[11];
cx q[54],q[11];
u3(0,0,0.25) q[11];
cx q[11],q[0];
u3(0,0,-0.125) q[0];
cx q[56],q[0];
cx q[60],q[0];
u3(0,0,0.125) q[0];
cx q[60],q[5];
cx q[53],q[5];
u3(0,0,0.125) q[5];
cx q[53],q[43];
u3(0,0,0.25) q[43];
cx q[66],q[49];
u3(0,0,-0.125) q[49];
cx q[66],q[59];
u3(0,0,-0.25) q[59];
cx q[59],q[12];
cx q[69],q[61];
cx q[64],q[61];
u3(0,0,0.125) q[61];
cx q[61],q[51];
u3(0,0,-0.25) q[51];
cx q[61],q[19];
cx q[61],q[51];
cx q[64],q[19];
u3(0,0,0.125) q[19];
cx q[41],q[19];
u3(0,0,0.125) q[19];
cx q[60],q[41];
cx q[60],q[53];
cx q[53],q[43];
u3(0,0,-0.125) q[43];
cx q[64],q[5];
u3(0,0,0.125) q[5];
cx q[43],q[5];
cx q[53],q[5];
cx q[64],q[41];
u3(0,0,0.125) q[41];
cx q[56],q[41];
cx q[41],q[19];
u3(0,0,-0.125) q[19];
cx q[41],q[29];
u3(0,0,-0.125) q[29];
cx q[41],q[29];
cx q[64],q[42];
u3(0,0,0.125) q[42];
cx q[64],q[42];
cx q[42],q[3];
u3(0,0,0.125) q[3];
cx q[68],q[19];
cx q[30],q[19];
u3(0,0,0.125) q[19];
cx q[29],q[19];
cx q[30],q[19];
u3(0,0,0.125) q[19];
cx q[56],q[30];
cx q[69],q[46];
cx q[67],q[46];
u3(0,0,0.125) q[46];
cx q[55],q[46];
u3(0,0,0.125) q[46];
cx q[46],q[23];
u3(0,0,-0.125) q[23];
cx q[55],q[46];
cx q[47],q[46];
u3(0,0,0.125) q[46];
cx q[67],q[50];
cx q[65],q[50];
u3(0,0,-0.125) q[50];
cx q[50],q[7];
u3(0,0,0.125) q[7];
cx q[50],q[47];
u3(0,0,-0.125) q[47];
cx q[47],q[32];
cx q[50],q[32];
cx q[55],q[7];
u3(0,0,-0.125) q[7];
cx q[65],q[39];
u3(0,0,0.125) q[39];
cx q[39],q[33];
cx q[65],q[33];
u3(0,0,0.125) q[33];
cx q[39],q[33];
cx q[65],q[32];
u3(0,0,-0.125) q[32];
cx q[57],q[32];
u3(0,0,0.125) q[32];
cx q[65],q[57];
cx q[67],q[55];
cx q[55],q[21];
u3(0,0,-0.125) q[21];
cx q[55],q[46];
u3(0,0,-0.125) q[46];
cx q[67],q[50];
cx q[69],q[38];
cx q[49],q[38];
u3(0,0,-0.125) q[38];
cx q[54],q[49];
cx q[49],q[45];
u3(0,0,-0.125) q[45];
cx q[69],q[12];
u3(0,0,-0.25) q[12];
cx q[54],q[12];
u3(0,0,-0.125) q[12];
cx q[12],q[11];
u3(0,0,0.25) q[11];
cx q[42],q[12];
u3(0,0,-0.125) q[12];
cx q[51],q[42];
u3(0,0,0.125) q[42];
cx q[54],q[49];
cx q[64],q[42];
u3(0,0,-0.25) q[42];
cx q[42],q[9];
u3(0,0,0.125) q[9];
cx q[42],q[9];
cx q[51],q[9];
cx q[64],q[5];
u3(0,0,0.125) q[5];
cx q[60],q[5];
u3(0,0,0.125) q[5];
cx q[43],q[5];
cx q[34],q[5];
u3(0,0,-0.125) q[5];
cx q[34],q[5];
cx q[20],q[5];
u3(0,0,0.125) q[5];
cx q[20],q[5];
cx q[24],q[5];
u3(0,0,-0.125) q[5];
cx q[24],q[5];
cx q[31],q[5];
u3(0,0,-0.125) q[5];
cx q[60],q[30];
cx q[30],q[16];
u3(0,0,-0.125) q[16];
cx q[64],q[9];
u3(0,0,-0.125) q[9];
cx q[42],q[9];
cx q[64],q[61];
cx q[61],q[32];
u3(0,0,0.125) q[32];
cx q[61],q[57];
u3(0,0,-0.125) q[57];
cx q[57],q[32];
cx q[61],q[32];
u3(0,0,0.125) q[32];
cx q[63],q[61];
cx q[61],q[32];
u3(0,0,0.25) q[32];
cx q[32],q[4];
cx q[61],q[48];
u3(0,0,0.125) q[48];
cx q[64],q[51];
cx q[69],q[8];
cx q[61],q[8];
u3(0,0,0.125) q[8];
cx q[61],q[48];
cx q[48],q[8];
u3(0,0,0.25) q[8];
cx q[48],q[4];
u3(0,0,0.125) q[4];
cx q[48],q[8];
cx q[62],q[8];
cx q[36],q[8];
u3(0,0,-0.125) q[8];
cx q[36],q[8];
cx q[37],q[8];
u3(0,0,-0.125) q[8];
cx q[63],q[4];
u3(0,0,-0.125) q[4];
cx q[32],q[4];
cx q[25],q[4];
u3(0,0,-0.125) q[4];
cx q[25],q[4];
cx q[23],q[4];
u3(0,0,0.125) q[4];
cx q[23],q[4];
cx q[54],q[25];
cx q[54],q[12];
cx q[63],q[59];
cx q[69],q[66];
cx q[7],q[4];
cx q[55],q[4];
u3(0,0,0.125) q[4];
cx q[55],q[30];
u3(0,0,0.125) q[30];
cx q[55],q[30];
cx q[49],q[30];
u3(0,0,-0.125) q[30];
cx q[49],q[30];
cx q[66],q[30];
u3(0,0,-0.125) q[30];
cx q[38],q[30];
u3(0,0,0.125) q[30];
cx q[47],q[38];
cx q[66],q[25];
u3(0,0,0.125) q[25];
cx q[66],q[59];
u3(0,0,0.125) q[59];
cx q[59],q[32];
u3(0,0,0.125) q[32];
cx q[59],q[43];
cx q[60],q[43];
u3(0,0,-0.125) q[43];
cx q[60],q[53];
cx q[53],q[32];
u3(0,0,-0.125) q[32];
cx q[53],q[9];
u3(0,0,-0.125) q[9];
cx q[27],q[9];
u3(0,0,0.25) q[9];
cx q[53],q[27];
cx q[27],q[16];
u3(0,0,-0.125) q[16];
cx q[59],q[32];
cx q[60],q[56];
cx q[56],q[29];
cx q[63],q[43];
u3(0,0,-0.125) q[43];
cx q[43],q[24];
u3(0,0,-0.125) q[24];
cx q[63],q[32];
u3(0,0,0.125) q[32];
cx q[53],q[32];
cx q[63],q[62];
cx q[63],q[61];
cx q[64],q[29];
u3(0,0,0.125) q[29];
cx q[64],q[29];
cx q[66],q[38];
u3(0,0,-0.125) q[38];
cx q[50],q[38];
u3(0,0,0.125) q[38];
cx q[47],q[38];
u3(0,0,0.125) q[38];
cx q[50],q[17];
u3(0,0,0.25) q[17];
cx q[38],q[17];
u3(0,0,0.125) q[17];
cx q[54],q[50];
cx q[66],q[12];
u3(0,0,-0.125) q[12];
cx q[42],q[12];
cx q[66],q[49];
cx q[49],q[40];
u3(0,0,0.125) q[40];
cx q[51],q[40];
u3(0,0,-0.125) q[40];
cx q[55],q[49];
u3(0,0,-0.125) q[49];
cx q[49],q[7];
u3(0,0,-0.125) q[7];
cx q[55],q[49];
cx q[51],q[49];
u3(0,0,0.125) q[49];
cx q[49],q[7];
u3(0,0,-0.125) q[7];
cx q[52],q[49];
cx q[55],q[29];
u3(0,0,-0.125) q[29];
cx q[55],q[29];
cx q[29],q[21];
u3(0,0,-0.125) q[21];
cx q[55],q[21];
cx q[55],q[46];
cx q[46],q[18];
cx q[66],q[50];
u3(0,0,-0.125) q[50];
cx q[50],q[45];
u3(0,0,-0.125) q[45];
cx q[66],q[58];
cx q[58],q[27];
u3(0,0,-0.125) q[27];
cx q[61],q[58];
u3(0,0,0.125) q[58];
cx q[66],q[32];
u3(0,0,-0.25) q[32];
cx q[67],q[45];
u3(0,0,-0.125) q[45];
cx q[45],q[38];
cx q[67],q[26];
cx q[51],q[26];
u3(0,0,0.125) q[26];
cx q[62],q[51];
u3(0,0,-0.125) q[51];
cx q[51],q[36];
u3(0,0,0.125) q[36];
cx q[36],q[31];
cx q[51],q[37];
u3(0,0,0.125) q[37];
cx q[37],q[12];
u3(0,0,-0.125) q[12];
cx q[12],q[8];
u3(0,0,-0.125) q[8];
cx q[12],q[11];
cx q[62],q[7];
u3(0,0,0.125) q[7];
cx q[62],q[7];
cx q[51],q[7];
u3(0,0,-0.125) q[7];
cx q[51],q[49];
u3(0,0,0.125) q[49];
cx q[49],q[44];
cx q[51],q[37];
cx q[37],q[11];
u3(0,0,-0.125) q[11];
cx q[56],q[44];
u3(0,0,0.125) q[44];
cx q[62],q[7];
cx q[62],q[37];
cx q[37],q[14];
u3(0,0,0.125) q[14];
cx q[37],q[8];
u3(0,0,0.125) q[8];
cx q[37],q[12];
cx q[62],q[11];
cx q[62],q[51];
cx q[51],q[31];
u3(0,0,0.125) q[31];
cx q[62],q[44];
cx q[52],q[44];
u3(0,0,0.125) q[44];
cx q[49],q[44];
u3(0,0,-0.125) q[44];
cx q[49],q[40];
cx q[44],q[40];
u3(0,0,-0.125) q[40];
cx q[52],q[44];
cx q[44],q[34];
cx q[40],q[34];
u3(0,0,-0.125) q[34];
cx q[62],q[34];
cx q[34],q[5];
u3(0,0,-0.125) q[5];
cx q[34],q[31];
u3(0,0,-0.125) q[31];
cx q[67],q[12];
u3(0,0,-0.125) q[12];
cx q[53],q[12];
u3(0,0,-0.125) q[12];
cx q[67],q[12];
u3(0,0,0.125) q[12];
cx q[53],q[12];
cx q[63],q[12];
u3(0,0,0.125) q[12];
cx q[67],q[38];
cx q[38],q[17];
u3(0,0,0.125) q[17];
cx q[38],q[30];
u3(0,0,0.125) q[30];
cx q[9],q[3];
u3(0,0,0.25) q[3];
cx q[27],q[3];
u3(0,0,0.125) q[3];
cx q[42],q[27];
cx q[51],q[27];
u3(0,0,-0.125) q[27];
cx q[61],q[3];
u3(0,0,0.125) q[3];
cx q[58],q[3];
cx q[58],q[27];
cx q[58],q[34];
cx q[61],q[3];
u3(0,0,-0.125) q[3];
cx q[57],q[3];
u3(0,0,0.125) q[3];
cx q[57],q[51];
cx q[61],q[51];
u3(0,0,-0.125) q[51];
cx q[51],q[15];
cx q[61],q[27];
u3(0,0,0.125) q[27];
cx q[27],q[9];
cx q[42],q[9];
cx q[61],q[34];
u3(0,0,0.125) q[34];
cx q[61],q[35];
cx q[57],q[35];
cx q[35],q[25];
u3(0,0,0.375) q[25];
cx q[7],q[3];
u3(0,0,0.125) q[3];
cx q[11],q[7];
u3(0,0,-0.125) q[7];
cx q[54],q[11];
u3(0,0,0.125) q[11];
cx q[11],q[0];
cx q[54],q[0];
u3(0,0,0.125) q[0];
cx q[54],q[37];
cx q[56],q[0];
cx q[66],q[37];
u3(0,0,-0.125) q[37];
cx q[47],q[37];
cx q[66],q[11];
cx q[50],q[11];
u3(0,0,0.125) q[11];
cx q[50],q[37];
u3(0,0,0.125) q[37];
cx q[56],q[11];
cx q[66],q[32];
cx q[32],q[12];
u3(0,0,0.125) q[12];
cx q[32],q[29];
u3(0,0,0.25) q[29];
cx q[63],q[12];
u3(0,0,-0.125) q[12];
cx q[63],q[59];
cx q[66],q[54];
cx q[67],q[0];
u3(0,0,0.125) q[0];
cx q[67],q[0];
cx q[50],q[0];
u3(0,0,-0.125) q[0];
cx q[54],q[0];
u3(0,0,0.125) q[0];
cx q[54],q[50];
cx q[50],q[45];
u3(0,0,-0.125) q[45];
cx q[50],q[7];
u3(0,0,-0.125) q[7];
cx q[50],q[15];
u3(0,0,0.5) q[15];
cx q[36],q[15];
u3(0,0,-0.125) q[15];
cx q[50],q[11];
u3(0,0,0.125) q[11];
cx q[50],q[21];
u3(0,0,-0.125) q[21];
cx q[65],q[21];
u3(0,0,-0.125) q[21];
cx q[65],q[21];
cx q[47],q[21];
cx q[37],q[21];
u3(0,0,0.125) q[21];
cx q[37],q[18];
u3(0,0,0.125) q[18];
cx q[18],q[14];
cx q[34],q[14];
u3(0,0,-0.125) q[14];
cx q[67],q[45];
cx q[45],q[32];
u3(0,0,0.125) q[32];
cx q[32],q[29];
u3(0,0,-0.375) q[29];
cx q[64],q[29];
u3(0,0,-0.125) q[29];
cx q[38],q[29];
cx q[64],q[29];
u3(0,0,0.125) q[29];
cx q[68],q[45];
u3(0,0,-0.125) q[45];
cx q[68],q[32];
u3(0,0,-0.125) q[32];
cx q[68],q[45];
cx q[64],q[45];
u3(0,0,0.125) q[45];
cx q[64],q[45];
cx q[56],q[45];
u3(0,0,0.125) q[45];
cx q[45],q[38];
u3(0,0,0.25) q[38];
cx q[45],q[11];
cx q[45],q[41];
cx q[56],q[38];
cx q[38],q[30];
u3(0,0,0.125) q[30];
cx q[38],q[17];
u3(0,0,-0.125) q[17];
cx q[38],q[0];
cx q[50],q[0];
u3(0,0,-0.125) q[0];
cx q[56],q[45];
cx q[69],q[11];
u3(0,0,-0.125) q[11];
cx q[69],q[11];
cx q[54],q[11];
u3(0,0,-0.125) q[11];
cx q[11],q[7];
cx q[50],q[7];
cx q[50],q[47];
cx q[54],q[7];
u3(0,0,0.25) q[7];
cx q[41],q[7];
u3(0,0,0.125) q[7];
cx q[41],q[7];
cx q[33],q[7];
u3(0,0,0.125) q[7];
cx q[45],q[33];
u3(0,0,0.125) q[33];
cx q[54],q[11];
cx q[33],q[11];
u3(0,0,-0.125) q[11];
cx q[45],q[33];
cx q[33],q[0];
cx q[38],q[0];
u3(0,0,0.25) q[0];
cx q[54],q[21];
u3(0,0,0.125) q[21];
cx q[54],q[35];
u3(0,0,0.125) q[35];
cx q[54],q[25];
cx q[51],q[25];
u3(0,0,-0.125) q[25];
cx q[51],q[25];
cx q[51],q[9];
u3(0,0,0.125) q[9];
cx q[51],q[9];
cx q[36],q[9];
u3(0,0,-0.125) q[9];
cx q[54],q[35];
cx q[59],q[47];
u3(0,0,-0.125) q[47];
cx q[59],q[43];
cx q[59],q[47];
cx q[7],q[0];
cx q[45],q[0];
u3(0,0,-0.125) q[0];
cx q[45],q[41];
cx q[67],q[0];
u3(0,0,0.125) q[0];
cx q[4],q[0];
cx q[67],q[0];
u3(0,0,-0.125) q[0];
cx q[4],q[0];
cx q[67],q[2];
cx q[67],q[25];
u3(0,0,0.25) q[25];
cx q[53],q[25];
u3(0,0,-0.125) q[25];
cx q[67],q[25];
u3(0,0,0.125) q[25];
cx q[53],q[25];
cx q[25],q[10];
u3(0,0,-0.25) q[10];
cx q[41],q[25];
u3(0,0,-0.25) q[25];
cx q[67],q[10];
cx q[61],q[10];
u3(0,0,0.25) q[10];
cx q[61],q[10];
cx q[61],q[58];
cx q[58],q[14];
u3(0,0,0.125) q[14];
cx q[34],q[14];
u3(0,0,-0.125) q[14];
cx q[61],q[57];
cx q[57],q[51];
cx q[51],q[15];
u3(0,0,0.125) q[15];
cx q[65],q[10];
u3(0,0,-0.125) q[10];
cx q[65],q[10];
cx q[68],q[2];
cx q[66],q[2];
u3(0,0,-0.125) q[2];
cx q[38],q[2];
u3(0,0,-0.125) q[2];
cx q[38],q[10];
u3(0,0,-0.125) q[10];
cx q[38],q[10];
cx q[41],q[10];
u3(0,0,0.125) q[10];
cx q[66],q[2];
u3(0,0,-0.125) q[2];
cx q[68],q[19];
cx q[41],q[19];
u3(0,0,0.125) q[19];
cx q[41],q[25];
cx q[25],q[2];
cx q[38],q[2];
u3(0,0,0.125) q[2];
cx q[38],q[17];
cx q[68],q[9];
u3(0,0,-0.125) q[9];
cx q[68],q[9];
cx q[31],q[9];
u3(0,0,0.125) q[9];
cx q[45],q[9];
u3(0,0,0.125) q[9];
cx q[45],q[43];
u3(0,0,0.25) q[43];
cx q[45],q[43];
cx q[43],q[25];
u3(0,0,0.25) q[25];
cx q[43],q[24];
cx q[43],q[34];
cx q[43],q[25];
cx q[43],q[26];
cx q[45],q[37];
cx q[47],q[37];
u3(0,0,-0.125) q[37];
cx q[37],q[17];
u3(0,0,0.25) q[17];
cx q[40],q[17];
cx q[58],q[34];
u3(0,0,-0.125) q[34];
cx q[62],q[34];
u3(0,0,0.125) q[34];
cx q[62],q[44];
cx q[47],q[44];
u3(0,0,-0.25) q[44];
cx q[47],q[46];
cx q[46],q[41];
u3(0,0,0.125) q[41];
cx q[41],q[10];
u3(0,0,0.125) q[10];
cx q[46],q[2];
u3(0,0,0.125) q[2];
cx q[46],q[25];
u3(0,0,-0.125) q[25];
cx q[25],q[2];
cx q[46],q[2];
u3(0,0,0.125) q[2];
cx q[46],q[23];
cx q[23],q[13];
u3(0,0,-0.125) q[13];
cx q[46],q[41];
cx q[46],q[2];
cx q[46],q[25];
cx q[47],q[44];
cx q[56],q[23];
u3(0,0,0.125) q[23];
cx q[45],q[23];
u3(0,0,0.125) q[23];
cx q[56],q[23];
u3(0,0,-0.25) q[23];
cx q[23],q[7];
cx q[33],q[7];
u3(0,0,0.125) q[7];
cx q[55],q[7];
u3(0,0,-0.125) q[7];
cx q[55],q[7];
cx q[62],q[52];
cx q[63],q[13];
cx q[63],q[48];
cx q[48],q[2];
u3(0,0,-0.125) q[2];
cx q[66],q[13];
u3(0,0,-0.125) q[13];
cx q[58],q[13];
u3(0,0,0.125) q[13];
cx q[66],q[13];
u3(0,0,0.125) q[13];
cx q[58],q[13];
cx q[29],q[13];
u3(0,0,-0.125) q[13];
cx q[38],q[13];
u3(0,0,-0.125) q[13];
cx q[29],q[13];
u3(0,0,-0.125) q[13];
cx q[38],q[13];
cx q[38],q[29];
cx q[45],q[13];
u3(0,0,-0.125) q[13];
cx q[45],q[13];
cx q[41],q[13];
u3(0,0,-0.125) q[13];
cx q[45],q[37];
cx q[51],q[45];
u3(0,0,0.125) q[45];
cx q[45],q[42];
u3(0,0,0.125) q[42];
cx q[42],q[23];
u3(0,0,0.125) q[23];
cx q[51],q[25];
u3(0,0,-0.125) q[25];
cx q[51],q[29];
u3(0,0,-0.25) q[29];
cx q[51],q[26];
u3(0,0,0.125) q[26];
cx q[43],q[26];
cx q[51],q[46];
cx q[46],q[36];
u3(0,0,0.125) q[36];
cx q[46],q[36];
cx q[36],q[31];
cx q[36],q[15];
cx q[51],q[23];
cx q[45],q[23];
u3(0,0,0.125) q[23];
cx q[42],q[23];
cx q[45],q[31];
u3(0,0,-0.125) q[31];
cx q[31],q[9];
cx q[43],q[9];
u3(0,0,0.125) q[9];
cx q[45],q[31];
cx q[58],q[14];
cx q[48],q[14];
u3(0,0,0.125) q[14];
cx q[48],q[14];
cx q[35],q[14];
u3(0,0,0.125) q[14];
cx q[35],q[14];
cx q[37],q[14];
u3(0,0,0.125) q[14];
cx q[37],q[17];
u3(0,0,-0.125) q[17];
cx q[44],q[37];
u3(0,0,0.125) q[37];
cx q[44],q[17];
u3(0,0,0.125) q[17];
cx q[40],q[17];
u3(0,0,0.125) q[17];
cx q[44],q[17];
cx q[44],q[26];
u3(0,0,0.25) q[26];
cx q[44],q[26];
cx q[44],q[40];
cx q[48],q[17];
u3(0,0,-0.25) q[17];
cx q[48],q[17];
cx q[67],q[10];
u3(0,0,-0.125) q[10];
cx q[68],q[24];
u3(0,0,0.25) q[24];
cx q[68],q[24];
cx q[64],q[24];
u3(0,0,-0.125) q[24];
cx q[64],q[24];
cx q[65],q[24];
u3(0,0,0.25) q[24];
cx q[65],q[39];
cx q[39],q[13];
cx q[41],q[13];
u3(0,0,0.125) q[13];
cx q[41],q[19];
cx q[29],q[19];
u3(0,0,0.25) q[19];
cx q[26],q[19];
u3(0,0,-0.125) q[19];
cx q[51],q[19];
u3(0,0,-0.125) q[19];
cx q[26],q[19];
cx q[26],q[9];
u3(0,0,0.125) q[9];
cx q[43],q[9];
u3(0,0,-0.125) q[9];
cx q[26],q[9];
cx q[38],q[9];
u3(0,0,-0.25) q[9];
cx q[38],q[9];
cx q[17],q[9];
u3(0,0,-0.25) q[9];
cx q[17],q[9];
cx q[51],q[31];
cx q[52],q[9];
u3(0,0,-0.125) q[9];
cx q[53],q[19];
u3(0,0,0.125) q[19];
cx q[66],q[19];
u3(0,0,-0.125) q[19];
cx q[53],q[19];
u3(0,0,-0.125) q[19];
cx q[53],q[9];
u3(0,0,-0.125) q[9];
cx q[53],q[31];
u3(0,0,0.125) q[31];
cx q[59],q[31];
u3(0,0,-0.125) q[31];
cx q[53],q[31];
u3(0,0,-0.125) q[31];
cx q[53],q[9];
cx q[52],q[9];
cx q[52],q[49];
cx q[59],q[31];
cx q[33],q[31];
u3(0,0,0.125) q[31];
cx q[39],q[31];
u3(0,0,-0.125) q[31];
cx q[33],q[31];
u3(0,0,0.125) q[31];
cx q[39],q[31];
cx q[47],q[31];
u3(0,0,-0.125) q[31];
cx q[31],q[5];
u3(0,0,0.25) q[5];
cx q[47],q[31];
cx q[48],q[31];
u3(0,0,-0.125) q[31];
cx q[31],q[2];
u3(0,0,-0.125) q[2];
cx q[48],q[2];
u3(0,0,-0.125) q[2];
cx q[28],q[2];
cx q[48],q[2];
u3(0,0,0.125) q[2];
cx q[48],q[31];
cx q[31],q[17];
u3(0,0,0.125) q[17];
cx q[28],q[17];
u3(0,0,-0.125) q[17];
cx q[31],q[28];
cx q[58],q[28];
u3(0,0,-0.25) q[28];
cx q[58],q[28];
cx q[38],q[28];
u3(0,0,-0.25) q[28];
cx q[28],q[22];
cx q[28],q[2];
cx q[38],q[20];
cx q[22],q[20];
u3(0,0,0.125) q[20];
cx q[40],q[22];
u3(0,0,-0.125) q[22];
cx q[22],q[2];
u3(0,0,-0.25) q[2];
cx q[38],q[22];
cx q[43],q[20];
u3(0,0,0.125) q[20];
cx q[48],q[22];
u3(0,0,0.125) q[22];
cx q[48],q[22];
cx q[33],q[22];
u3(0,0,-0.125) q[22];
cx q[22],q[11];
u3(0,0,-0.125) q[11];
cx q[33],q[11];
u3(0,0,0.125) q[11];
cx q[33],q[22];
cx q[43],q[22];
u3(0,0,0.125) q[22];
cx q[22],q[20];
cx q[43],q[20];
u3(0,0,-0.125) q[20];
cx q[43],q[20];
cx q[20],q[4];
u3(0,0,0.125) q[4];
cx q[43],q[22];
cx q[51],q[40];
cx q[40],q[29];
u3(0,0,0.125) q[29];
cx q[40],q[25];
u3(0,0,-0.25) q[25];
cx q[25],q[15];
cx q[49],q[29];
cx q[51],q[45];
cx q[45],q[42];
cx q[45],q[3];
cx q[51],q[45];
cx q[55],q[49];
cx q[65],q[19];
cx q[66],q[19];
u3(0,0,0.125) q[19];
cx q[24],q[19];
u3(0,0,0.125) q[19];
cx q[66],q[6];
cx q[68],q[37];
u3(0,0,0.125) q[37];
cx q[44],q[37];
cx q[68],q[37];
cx q[37],q[31];
u3(0,0,0.125) q[31];
cx q[45],q[37];
u3(0,0,-0.125) q[37];
cx q[45],q[37];
cx q[37],q[18];
cx q[51],q[45];
cx q[51],q[40];
cx q[51],q[46];
cx q[51],q[36];
cx q[60],q[18];
u3(0,0,-0.125) q[18];
cx q[41],q[18];
u3(0,0,-0.125) q[18];
cx q[60],q[18];
u3(0,0,0.125) q[18];
cx q[41],q[18];
cx q[69],q[19];
u3(0,0,-0.125) q[19];
cx q[7],q[0];
u3(0,0,-0.125) q[0];
cx q[20],q[0];
u3(0,0,0.125) q[0];
cx q[35],q[20];
u3(0,0,0.125) q[20];
cx q[35],q[20];
cx q[23],q[20];
u3(0,0,0.125) q[20];
cx q[20],q[7];
u3(0,0,0.125) q[7];
cx q[35],q[0];
cx q[23],q[0];
u3(0,0,0.125) q[0];
cx q[23],q[20];
cx q[20],q[0];
u3(0,0,0.125) q[0];
cx q[7],q[4];
cx q[35],q[4];
u3(0,0,0.125) q[4];
cx q[35],q[4];
cx q[35],q[0];
cx q[38],q[4];
u3(0,0,0.125) q[4];
cx q[38],q[4];
cx q[39],q[0];
u3(0,0,0.125) q[0];
cx q[13],q[0];
u3(0,0,-0.125) q[0];
cx q[39],q[0];
u3(0,0,-0.125) q[0];
cx q[13],q[0];
cx q[57],q[0];
u3(0,0,0.125) q[0];
cx q[57],q[3];
u3(0,0,-0.125) q[3];
cx q[58],q[4];
u3(0,0,0.25) q[4];
cx q[58],q[4];
cx q[47],q[4];
u3(0,0,-0.125) q[4];
cx q[47],q[20];
cx q[20],q[4];
u3(0,0,-0.125) q[4];
cx q[20],q[5];
u3(0,0,0.125) q[5];
cx q[47],q[6];
cx q[47],q[31];
u3(0,0,0.125) q[31];
cx q[58],q[42];
u3(0,0,0.125) q[42];
cx q[58],q[42];
cx q[54],q[42];
u3(0,0,-0.125) q[42];
cx q[61],q[6];
u3(0,0,0.125) q[6];
cx q[48],q[6];
u3(0,0,0.125) q[6];
cx q[61],q[6];
u3(0,0,-0.125) q[6];
cx q[48],q[6];
cx q[48],q[38];
cx q[56],q[6];
u3(0,0,-0.125) q[6];
cx q[39],q[6];
u3(0,0,-0.125) q[6];
cx q[56],q[6];
u3(0,0,-0.125) q[6];
cx q[39],q[6];
cx q[33],q[6];
u3(0,0,-0.25) q[6];
cx q[39],q[13];
cx q[57],q[6];
u3(0,0,0.125) q[6];
cx q[33],q[6];
u3(0,0,0.125) q[6];
cx q[33],q[18];
u3(0,0,0.125) q[18];
cx q[69],q[18];
u3(0,0,0.125) q[18];
cx q[69],q[19];
cx q[7],q[3];
cx q[20],q[3];
cx q[47],q[3];
u3(0,0,0.125) q[3];
cx q[9],q[0];
u3(0,0,0.125) q[0];
cx q[13],q[9];
u3(0,0,-0.125) q[9];
cx q[45],q[9];
u3(0,0,-0.125) q[9];
cx q[45],q[3];
cx q[45],q[30];
cx q[38],q[30];
u3(0,0,-0.125) q[30];
cx q[38],q[28];
cx q[47],q[3];
u3(0,0,0.25) q[3];
cx q[47],q[3];
cx q[57],q[0];
u3(0,0,-0.125) q[0];
cx q[57],q[6];
cx q[22],q[6];
u3(0,0,-0.125) q[6];
cx q[30],q[22];
u3(0,0,-0.125) q[22];
cx q[30],q[16];
cx q[28],q[16];
u3(0,0,-0.125) q[16];
cx q[28],q[16];
cx q[48],q[22];
u3(0,0,0.125) q[22];
cx q[48],q[16];
cx q[48],q[30];
cx q[6],q[2];
u3(0,0,-0.25) q[2];
cx q[65],q[16];
u3(0,0,-0.125) q[16];
cx q[65],q[16];
cx q[62],q[16];
u3(0,0,0.125) q[16];
cx q[62],q[16];
cx q[23],q[16];
u3(0,0,0.125) q[16];
cx q[23],q[16];
cx q[62],q[34];
cx q[43],q[34];
cx q[65],q[24];
cx q[67],q[3];
u3(0,0,0.125) q[3];
cx q[63],q[3];
u3(0,0,0.125) q[3];
cx q[67],q[3];
u3(0,0,-0.25) q[3];
cx q[63],q[3];
cx q[59],q[3];
u3(0,0,0.125) q[3];
cx q[59],q[3];
cx q[26],q[3];
u3(0,0,0.125) q[3];
cx q[26],q[13];
cx q[13],q[3];
u3(0,0,0.125) q[3];
cx q[26],q[13];
cx q[59],q[21];
cx q[37],q[21];
cx q[37],q[14];
cx q[37],q[8];
cx q[37],q[31];
cx q[54],q[21];
u3(0,0,-0.125) q[21];
cx q[47],q[21];
u3(0,0,-0.125) q[21];
cx q[21],q[1];
cx q[47],q[20];
cx q[20],q[16];
cx q[16],q[4];
u3(0,0,0.125) q[4];
cx q[16],q[7];
u3(0,0,0.25) q[7];
cx q[20],q[5];
cx q[20],q[16];
cx q[40],q[5];
u3(0,0,0.25) q[5];
cx q[24],q[5];
cx q[24],q[4];
cx q[16],q[4];
u3(0,0,0.25) q[4];
cx q[16],q[7];
cx q[40],q[5];
u3(0,0,-0.125) q[5];
cx q[40],q[15];
u3(0,0,-0.25) q[15];
cx q[40],q[29];
u3(0,0,0.125) q[29];
cx q[40],q[25];
cx q[30],q[25];
u3(0,0,-0.125) q[25];
cx q[45],q[29];
u3(0,0,0.125) q[29];
cx q[46],q[7];
u3(0,0,-0.125) q[7];
cx q[41],q[7];
u3(0,0,-0.125) q[7];
cx q[41],q[7];
cx q[46],q[4];
cx q[24],q[4];
u3(0,0,-0.125) q[4];
cx q[24],q[19];
cx q[19],q[16];
u3(0,0,0.125) q[16];
cx q[19],q[13];
u3(0,0,-0.125) q[13];
cx q[19],q[13];
cx q[24],q[5];
cx q[49],q[30];
u3(0,0,-0.125) q[30];
cx q[49],q[30];
cx q[30],q[22];
cx q[54],q[42];
cx q[55],q[49];
cx q[56],q[30];
cx q[30],q[19];
u3(0,0,0.125) q[19];
cx q[30],q[19];
cx q[56],q[30];
cx q[30],q[27];
cx q[42],q[27];
u3(0,0,-0.125) q[27];
cx q[65],q[16];
u3(0,0,0.125) q[16];
cx q[67],q[10];
cx q[10],q[7];
u3(0,0,0.125) q[7];
cx q[10],q[6];
cx q[13],q[10];
cx q[22],q[7];
u3(0,0,0.125) q[7];
cx q[34],q[13];
u3(0,0,-0.125) q[13];
cx q[34],q[19];
u3(0,0,-0.125) q[19];
cx q[19],q[13];
cx q[34],q[13];
u3(0,0,-0.125) q[13];
cx q[19],q[13];
cx q[36],q[34];
u3(0,0,0.125) q[34];
cx q[36],q[34];
cx q[34],q[5];
u3(0,0,-0.25) q[5];
cx q[46],q[6];
u3(0,0,-0.125) q[6];
cx q[22],q[6];
u3(0,0,0.125) q[6];
cx q[46],q[10];
u3(0,0,0.125) q[10];
cx q[41],q[10];
u3(0,0,0.125) q[10];
cx q[41],q[10];
cx q[46],q[4];
cx q[14],q[4];
u3(0,0,0.125) q[4];
cx q[48],q[4];
u3(0,0,0.125) q[4];
cx q[14],q[4];
cx q[68],q[3];
u3(0,0,-0.125) q[3];
cx q[7],q[6];
cx q[22],q[6];
u3(0,0,0.125) q[6];
cx q[22],q[4];
cx q[33],q[6];
u3(0,0,0.125) q[6];
cx q[33],q[11];
cx q[22],q[11];
cx q[23],q[11];
u3(0,0,0.125) q[11];
cx q[33],q[6];
cx q[33],q[18];
cx q[48],q[4];
u3(0,0,0.125) q[4];
cx q[48],q[38];
cx q[38],q[4];
u3(0,0,0.125) q[4];
cx q[48],q[28];
cx q[69],q[18];
cx q[9],q[0];
cx q[45],q[0];
cx q[10],q[0];
u3(0,0,-0.125) q[0];
cx q[13],q[10];
cx q[13],q[3];
cx q[13],q[9];
cx q[14],q[10];
u3(0,0,0.125) q[10];
cx q[35],q[10];
u3(0,0,0.125) q[10];
cx q[35],q[10];
cx q[14],q[10];
cx q[22],q[10];
u3(0,0,0.25) q[10];
cx q[22],q[10];
cx q[10],q[7];
u3(0,0,-0.125) q[7];
cx q[14],q[7];
cx q[22],q[7];
u3(0,0,0.25) q[7];
cx q[14],q[7];
cx q[22],q[4];
cx q[38],q[0];
u3(0,0,-0.125) q[0];
cx q[42],q[10];
u3(0,0,-0.25) q[10];
cx q[42],q[10];
cx q[30],q[10];
u3(0,0,0.25) q[10];
cx q[30],q[10];
cx q[45],q[10];
cx q[68],q[3];
cx q[34],q[3];
u3(0,0,0.25) q[3];
cx q[34],q[19];
cx q[5],q[3];
u3(0,0,0.25) q[3];
cx q[59],q[3];
u3(0,0,0.125) q[3];
cx q[59],q[3];
cx q[59],q[1];
cx q[63],q[3];
u3(0,0,0.125) q[3];
cx q[5],q[3];
cx q[26],q[3];
cx q[34],q[3];
cx q[34],q[5];
cx q[42],q[3];
cx q[63],q[3];
u3(0,0,-0.125) q[3];
cx q[26],q[3];
cx q[42],q[3];
cx q[66],q[1];
cx q[68],q[32];
cx q[32],q[10];
u3(0,0,-0.125) q[10];
cx q[32],q[12];
cx q[12],q[8];
u3(0,0,-0.125) q[8];
cx q[45],q[12];
cx q[35],q[12];
u3(0,0,0.25) q[12];
cx q[35],q[12];
cx q[45],q[8];
cx q[40],q[8];
u3(0,0,0.125) q[8];
cx q[40],q[8];
cx q[46],q[8];
u3(0,0,0.125) q[8];
cx q[36],q[8];
u3(0,0,0.125) q[8];
cx q[36],q[8];
cx q[46],q[8];
cx q[57],q[12];
u3(0,0,0.125) q[12];
cx q[23],q[12];
u3(0,0,-0.125) q[12];
cx q[57],q[12];
u3(0,0,-0.125) q[12];
cx q[23],q[12];
cx q[25],q[12];
u3(0,0,0.125) q[12];
cx q[30],q[12];
u3(0,0,0.125) q[12];
cx q[25],q[12];
u3(0,0,0.125) q[12];
cx q[12],q[7];
cx q[30],q[7];
u3(0,0,-0.125) q[7];
cx q[30],q[25];
cx q[31],q[30];
cx q[47],q[30];
cx q[30],q[12];
u3(0,0,-0.25) q[12];
cx q[30],q[27];
cx q[47],q[1];
cx q[49],q[7];
u3(0,0,-0.125) q[7];
cx q[12],q[7];
cx q[49],q[29];
cx q[29],q[8];
u3(0,0,0.125) q[8];
cx q[45],q[8];
u3(0,0,-0.25) q[8];
cx q[40],q[8];
u3(0,0,-0.125) q[8];
cx q[49],q[7];
cx q[64],q[1];
u3(0,0,-0.125) q[1];
cx q[24],q[1];
u3(0,0,-0.125) q[1];
cx q[64],q[1];
cx q[43],q[1];
u3(0,0,0.125) q[1];
cx q[24],q[1];
u3(0,0,-0.125) q[1];
cx q[43],q[1];
cx q[53],q[1];
u3(0,0,0.125) q[1];
cx q[53],q[1];
cx q[63],q[1];
u3(0,0,0.125) q[1];
cx q[63],q[1];
cx q[65],q[10];
u3(0,0,-0.125) q[10];
cx q[10],q[0];
cx q[32],q[0];
cx q[65],q[16];
cx q[16],q[11];
cx q[19],q[11];
u3(0,0,-0.125) q[11];
cx q[19],q[15];
cx q[19],q[16];
cx q[23],q[11];
cx q[25],q[15];
u3(0,0,-0.125) q[15];
cx q[25],q[4];
cx q[38],q[4];
u3(0,0,0.125) q[4];
cx q[25],q[4];
cx q[38],q[1];
u3(0,0,-0.25) q[1];
cx q[38],q[1];
cx q[38],q[0];
cx q[4],q[1];
u3(0,0,-0.125) q[1];
cx q[45],q[16];
cx q[16],q[9];
u3(0,0,-0.25) q[9];
cx q[47],q[1];
u3(0,0,0.125) q[1];
cx q[4],q[1];
cx q[31],q[1];
cx q[27],q[1];
u3(0,0,-0.125) q[1];
cx q[27],q[7];
cx q[31],q[27];
cx q[47],q[27];
cx q[47],q[31];
cx q[31],q[30];
cx q[31],q[12];
cx q[47],q[21];
cx q[51],q[15];
u3(0,0,-0.125) q[15];
cx q[19],q[15];
cx q[51],q[15];
cx q[53],q[1];
u3(0,0,-0.125) q[1];
cx q[53],q[1];
cx q[59],q[21];
cx q[62],q[11];
u3(0,0,-0.125) q[11];
cx q[65],q[0];
cx q[16],q[0];
u3(0,0,0.125) q[0];
cx q[16],q[2];
cx q[4],q[0];
u3(0,0,-0.125) q[0];
cx q[4],q[0];
cx q[45],q[2];
cx q[7],q[1];
u3(0,0,-0.25) q[1];
cx q[7],q[6];
u3(0,0,-0.25) q[6];
cx q[27],q[6];
cx q[6],q[2];
u3(0,0,0.125) q[2];
cx q[17],q[6];
cx q[28],q[2];
u3(0,0,-0.125) q[2];
cx q[28],q[6];
u3(0,0,0.25) q[6];
cx q[28],q[17];
cx q[17],q[6];
cx q[7],q[1];
cx q[8],q[1];
cx q[27],q[1];
cx q[27],q[7];
cx q[29],q[1];
cx q[40],q[1];
cx q[45],q[1];
u3(0,0,-0.125) q[1];
cx q[45],q[29];
cx q[29],q[8];
cx q[40],q[8];
cx q[45],q[16];
cx q[16],q[11];
cx q[16],q[9];
cx q[16],q[2];
cx q[16],q[0];
cx q[28],q[2];
cx q[45],q[32];
cx q[32],q[10];
cx q[62],q[11];
cx q[63],q[1];
u3(0,0,-0.125) q[1];
cx q[65],q[10];
cx q[8],q[1];
cx q[63],q[1];
