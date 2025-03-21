OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
rz(272.3125) q[23];
rz(271.625) q[22];
rz(-154.625) q[21];
rz(272.3125) q[20];
rz(271.625) q[19];
rz(-154.625) q[18];
rz(272.3125) q[17];
rz(271.625) q[16];
rz(-154.625) q[15];
rz(272.3125) q[14];
rz(271.625) q[13];
rz(-154.625) q[12];
rz(272.3125) q[11];
rz(271.625) q[10];
rz(-154.625) q[9];
rz(272.3125) q[8];
rz(271.625) q[7];
rz(-154.625) q[6];
rz(272.3125) q[5];
rz(271.625) q[4];
rz(-154.625) q[3];
rz(272.3125) q[2];
rz(271.625) q[1];
rz(-154.625) q[0];
cx q[23],q[22];
rz(311.4375) q[22];
cx q[23],q[21];
rz(-125.6875) q[21];
cx q[23],q[20];
rz(-120.375) q[20];
cx q[23],q[19];
rz(-4.03125) q[19];
cx q[23],q[18];
rz(2.34375) q[18];
cx q[23],q[2];
rz(-120.375) q[2];
cx q[23],q[1];
rz(-4.03125) q[1];
cx q[23],q[0];
rz(2.34375) q[0];
cx q[17],q[16];
rz(311.4375) q[16];
cx q[17],q[15];
rz(-125.6875) q[15];
cx q[17],q[14];
rz(-120.375) q[14];
cx q[17],q[13];
rz(-4.03125) q[13];
cx q[17],q[12];
rz(2.34375) q[12];
cx q[11],q[10];
rz(311.4375) q[10];
cx q[11],q[9];
rz(-125.6875) q[9];
cx q[11],q[8];
rz(-120.375) q[8];
cx q[11],q[7];
rz(-4.03125) q[7];
cx q[11],q[6];
rz(2.34375) q[6];
cx q[5],q[4];
rz(311.4375) q[4];
cx q[5],q[3];
rz(-125.6875) q[3];
cx q[22],q[21];
rz(-85.625) q[21];
cx q[22],q[20];
rz(-4.03125) q[20];
cx q[22],q[19];
rz(-133.8125) q[19];
cx q[22],q[18];
rz(34.625) q[18];
cx q[22],q[2];
rz(-4.03125) q[2];
cx q[22],q[1];
rz(-133.8125) q[1];
cx q[22],q[0];
rz(34.625) q[0];
cx q[16],q[15];
rz(-85.625) q[15];
cx q[16],q[14];
rz(-4.03125) q[14];
cx q[16],q[13];
rz(-133.8125) q[13];
cx q[16],q[12];
rz(34.625) q[12];
cx q[10],q[9];
rz(-85.625) q[9];
cx q[10],q[8];
rz(-4.03125) q[8];
cx q[10],q[7];
rz(-133.8125) q[7];
cx q[10],q[6];
rz(34.625) q[6];
cx q[4],q[3];
rz(-85.625) q[3];
cx q[23],q[21];
rz(-264.3125) q[21];
cx q[23],q[20];
rz(-63.125) q[20];
cx q[23],q[19];
rz(3.03125) q[19];
cx q[23],q[18];
rz(24.84375) q[18];
cx q[23],q[2];
rz(-63.125) q[2];
cx q[23],q[1];
rz(3.03125) q[1];
cx q[23],q[0];
rz(24.84375) q[0];
cx q[17],q[15];
rz(-264.3125) q[15];
cx q[17],q[14];
rz(-63.125) q[14];
cx q[17],q[13];
rz(3.03125) q[13];
cx q[17],q[12];
rz(24.84375) q[12];
cx q[11],q[9];
rz(-264.3125) q[9];
cx q[11],q[8];
rz(-63.125) q[8];
cx q[11],q[7];
rz(3.03125) q[7];
cx q[11],q[6];
rz(24.84375) q[6];
cx q[5],q[3];
rz(-264.3125) q[3];
cx q[21],q[20];
rz(2.34375) q[20];
cx q[21],q[19];
rz(34.625) q[19];
cx q[21],q[18];
rz(-92.5625) q[18];
cx q[21],q[2];
rz(2.34375) q[2];
cx q[21],q[1];
rz(34.625) q[1];
cx q[21],q[0];
rz(-92.5625) q[0];
cx q[15],q[14];
rz(2.34375) q[14];
cx q[15],q[13];
rz(34.625) q[13];
cx q[15],q[12];
rz(-92.5625) q[12];
cx q[9],q[8];
rz(2.34375) q[8];
cx q[9],q[7];
rz(34.625) q[7];
cx q[9],q[6];
rz(-92.5625) q[6];
cx q[23],q[20];
rz(28.1875) q[20];
cx q[23],q[19];
rz(62.65625) q[19];
cx q[23],q[18];
rz(-11.15625) q[18];
cx q[23],q[2];
rz(28.1875) q[2];
cx q[23],q[1];
rz(62.65625) q[1];
cx q[23],q[0];
rz(-11.15625) q[0];
cx q[17],q[14];
rz(28.1875) q[14];
cx q[17],q[13];
rz(62.65625) q[13];
cx q[17],q[12];
rz(-11.15625) q[12];
cx q[11],q[8];
rz(28.1875) q[8];
cx q[11],q[7];
rz(62.65625) q[7];
cx q[11],q[6];
rz(-11.15625) q[6];
cx q[22],q[20];
rz(44.65625) q[20];
cx q[22],q[19];
rz(51.375) q[19];
cx q[22],q[18];
rz(20.875) q[18];
cx q[22],q[2];
rz(44.65625) q[2];
cx q[22],q[1];
rz(51.375) q[1];
cx q[22],q[0];
rz(20.875) q[0];
cx q[16],q[14];
rz(44.65625) q[14];
cx q[16],q[13];
rz(51.375) q[13];
cx q[16],q[12];
rz(20.875) q[12];
cx q[10],q[8];
rz(44.65625) q[8];
cx q[10],q[7];
rz(51.375) q[7];
cx q[10],q[6];
rz(20.875) q[6];
cx q[23],q[20];
rz(19.3125) q[20];
cx q[23],q[19];
rz(83.71875) q[19];
cx q[23],q[18];
rz(-25.03125) q[18];
cx q[23],q[2];
rz(19.3125) q[2];
cx q[23],q[1];
rz(83.71875) q[1];
cx q[23],q[0];
rz(-25.03125) q[0];
cx q[17],q[14];
rz(19.3125) q[14];
cx q[17],q[13];
rz(83.71875) q[13];
cx q[17],q[12];
rz(-25.03125) q[12];
cx q[11],q[8];
rz(19.3125) q[8];
cx q[11],q[7];
rz(83.71875) q[7];
cx q[11],q[6];
rz(-25.03125) q[6];
cx q[20],q[19];
rz(311.4375) q[19];
cx q[20],q[18];
rz(-125.6875) q[18];
cx q[2],q[1];
rz(311.4375) q[1];
cx q[2],q[0];
rz(-125.6875) q[0];
cx q[14],q[13];
rz(311.4375) q[13];
cx q[14],q[12];
rz(-125.6875) q[12];
cx q[8],q[7];
rz(311.4375) q[7];
cx q[8],q[6];
rz(-125.6875) q[6];
cx q[23],q[19];
rz(-63.125) q[19];
cx q[23],q[18];
rz(28.1875) q[18];
cx q[23],q[1];
rz(-63.125) q[1];
cx q[23],q[0];
rz(28.1875) q[0];
cx q[17],q[13];
rz(-63.125) q[13];
cx q[17],q[12];
rz(28.1875) q[12];
cx q[11],q[7];
rz(-63.125) q[7];
cx q[11],q[6];
rz(28.1875) q[6];
cx q[22],q[19];
rz(3.03125) q[19];
cx q[22],q[18];
rz(62.65625) q[18];
cx q[22],q[1];
rz(3.03125) q[1];
cx q[22],q[0];
rz(62.65625) q[0];
cx q[16],q[13];
rz(3.03125) q[13];
cx q[16],q[12];
rz(62.65625) q[12];
cx q[10],q[7];
rz(3.03125) q[7];
cx q[10],q[6];
rz(62.65625) q[6];
cx q[23],q[19];
rz(-102.75) q[19];
cx q[23],q[18];
rz(-11.125) q[18];
cx q[23],q[1];
rz(-102.75) q[1];
cx q[23],q[0];
rz(-11.125) q[0];
cx q[17],q[13];
rz(-102.75) q[13];
cx q[17],q[12];
rz(-11.125) q[12];
cx q[11],q[7];
rz(-102.75) q[7];
cx q[11],q[6];
rz(-11.125) q[6];
cx q[21],q[19];
rz(24.84375) q[19];
cx q[21],q[18];
rz(-11.15625) q[18];
cx q[21],q[1];
rz(24.84375) q[1];
cx q[21],q[0];
rz(-11.15625) q[0];
cx q[15],q[13];
rz(24.84375) q[13];
cx q[15],q[12];
rz(-11.15625) q[12];
cx q[9],q[7];
rz(24.84375) q[7];
cx q[9],q[6];
rz(-11.15625) q[6];
cx q[23],q[19];
rz(-11.125) q[19];
cx q[23],q[18];
rz(-126.125) q[18];
cx q[23],q[1];
rz(-11.125) q[1];
cx q[23],q[0];
rz(-126.125) q[0];
cx q[17],q[13];
rz(-11.125) q[13];
cx q[17],q[12];
rz(-126.125) q[12];
cx q[11],q[7];
rz(-11.125) q[7];
cx q[11],q[6];
rz(-126.125) q[6];
cx q[22],q[19];
rz(-23.21875) q[19];
cx q[22],q[18];
rz(-126.65625) q[18];
cx q[22],q[1];
rz(-23.21875) q[1];
cx q[22],q[0];
rz(-126.65625) q[0];
cx q[16],q[13];
rz(-23.21875) q[13];
cx q[16],q[12];
rz(-126.65625) q[12];
cx q[10],q[7];
rz(-23.21875) q[7];
cx q[10],q[6];
rz(-126.65625) q[6];
cx q[23],q[19];
rz(-2.25) q[19];
cx q[23],q[18];
rz(-93.5625) q[18];
cx q[23],q[1];
rz(-2.25) q[1];
cx q[23],q[0];
rz(-93.5625) q[0];
cx q[17],q[13];
rz(-2.25) q[13];
cx q[17],q[12];
rz(-93.5625) q[12];
cx q[11],q[7];
rz(-2.25) q[7];
cx q[11],q[6];
rz(-93.5625) q[6];
cx q[19],q[18];
rz(-85.625) q[18];
cx q[1],q[0];
rz(-85.625) q[0];
cx q[13],q[12];
rz(-85.625) q[12];
cx q[7],q[6];
rz(-85.625) q[6];
cx q[23],q[18];
rz(44.65625) q[18];
cx q[23],q[0];
rz(44.65625) q[0];
cx q[17],q[12];
rz(44.65625) q[12];
cx q[11],q[6];
rz(44.65625) q[6];
cx q[22],q[18];
rz(51.375) q[18];
cx q[22],q[0];
rz(51.375) q[0];
cx q[16],q[12];
rz(51.375) q[12];
cx q[10],q[6];
rz(51.375) q[6];
cx q[23],q[18];
rz(-23.21875) q[18];
cx q[23],q[0];
rz(-23.21875) q[0];
cx q[17],q[12];
rz(-23.21875) q[12];
cx q[11],q[6];
rz(-23.21875) q[6];
cx q[21],q[18];
rz(20.875) q[18];
cx q[21],q[0];
rz(20.875) q[0];
cx q[15],q[12];
rz(20.875) q[12];
cx q[9],q[6];
rz(20.875) q[6];
cx q[23],q[18];
rz(-126.65625) q[18];
cx q[23],q[0];
rz(-126.65625) q[0];
cx q[17],q[12];
rz(-126.65625) q[12];
cx q[11],q[6];
rz(-126.65625) q[6];
cx q[22],q[18];
rz(-254.5625) q[18];
cx q[22],q[0];
rz(-254.5625) q[0];
cx q[16],q[12];
rz(-254.5625) q[12];
cx q[10],q[6];
rz(-254.5625) q[6];
cx q[23],q[18];
rz(-128.03125) q[18];
cx q[23],q[0];
rz(-128.03125) q[0];
cx q[17],q[12];
rz(-128.03125) q[12];
cx q[11],q[6];
rz(-128.03125) q[6];
cx q[20],q[18];
rz(-264.3125) q[18];
cx q[2],q[0];
rz(-264.3125) q[0];
cx q[14],q[12];
rz(-264.3125) q[12];
cx q[8],q[6];
rz(-264.3125) q[6];
cx q[23],q[18];
rz(19.3125) q[18];
cx q[23],q[0];
rz(19.3125) q[0];
cx q[17],q[12];
rz(19.3125) q[12];
cx q[11],q[6];
rz(19.3125) q[6];
cx q[22],q[18];
rz(83.71875) q[18];
cx q[22],q[0];
rz(83.71875) q[0];
cx q[16],q[12];
rz(83.71875) q[12];
cx q[10],q[6];
rz(83.71875) q[6];
cx q[23],q[18];
rz(-2.25) q[18];
cx q[23],q[0];
rz(-2.25) q[0];
cx q[17],q[12];
rz(-2.25) q[12];
cx q[11],q[6];
rz(-2.25) q[6];
cx q[21],q[18];
rz(-25.03125) q[18];
cx q[21],q[0];
rz(-25.03125) q[0];
cx q[15],q[12];
rz(-25.03125) q[12];
cx q[9],q[6];
rz(-25.03125) q[6];
cx q[23],q[18];
rz(-93.5625) q[18];
cx q[23],q[0];
rz(-93.5625) q[0];
cx q[17],q[12];
rz(-93.5625) q[12];
cx q[11],q[6];
rz(-93.5625) q[6];
cx q[22],q[18];
rz(-128.03125) q[18];
cx q[22],q[0];
rz(-128.03125) q[0];
cx q[16],q[12];
rz(-128.03125) q[12];
cx q[10],q[6];
rz(-128.03125) q[6];
cx q[23],q[18];
rz(-163.625) q[18];
cx q[23],q[0];
rz(-163.625) q[0];
cx q[17],q[12];
rz(-163.625) q[12];
cx q[11],q[6];
rz(-163.625) q[6];
cx q[21],q[20];
cx q[20],q[17];
rz(-120.375) q[17];
cx q[20],q[16];
rz(-63.125) q[16];
cx q[20],q[15];
rz(19.3125) q[15];
cx q[20],q[15];
cx q[17],q[15];
rz(44.65625) q[15];
cx q[17],q[16];
cx q[16],q[15];
rz(2.34375) q[15];
cx q[20],q[16];
rz(-4.03125) q[16];
cx q[21],q[2];
cx q[5],q[2];
rz(-120.375) q[2];
cx q[3],q[2];
rz(44.65625) q[2];
cx q[5],q[2];
rz(19.3125) q[2];
cx q[4],q[2];
rz(2.34375) q[2];
cx q[5],q[2];
rz(28.1875) q[2];
cx q[3],q[2];
rz(-4.03125) q[2];
cx q[5],q[2];
rz(-63.125) q[2];
cx q[9],q[8];
cx q[8],q[5];
rz(-120.375) q[5];
cx q[8],q[4];
rz(-63.125) q[4];
cx q[8],q[3];
rz(19.3125) q[3];
cx q[8],q[3];
cx q[5],q[3];
rz(44.65625) q[3];
cx q[5],q[4];
cx q[4],q[3];
rz(2.34375) q[3];
cx q[8],q[4];
rz(-4.03125) q[4];
cx q[14],q[13];
cx q[13],q[11];
rz(-4.03125) q[11];
cx q[13],q[10];
rz(3.03125) q[10];
cx q[13],q[9];
rz(83.71875) q[9];
cx q[13],q[9];
cx q[11],q[9];
rz(51.375) q[9];
cx q[11],q[10];
cx q[10],q[9];
rz(34.625) q[9];
cx q[13],q[10];
rz(-133.8125) q[10];
cx q[21],q[17];
cx q[19],q[17];
rz(-4.03125) q[17];
cx q[20],q[17];
rz(-63.125) q[17];
cx q[17],q[16];
rz(3.03125) q[16];
cx q[17],q[15];
rz(62.65625) q[15];
cx q[20],q[16];
rz(-102.75) q[16];
cx q[20],q[15];
rz(-11.125) q[15];
cx q[21],q[2];
cx q[2],q[1];
rz(3.03125) q[1];
cx q[2],q[0];
rz(-23.21875) q[0];
cx q[7],q[6];
cx q[6],q[5];
rz(28.1875) q[5];
cx q[6],q[3];
rz(-11.15625) q[3];
cx q[6],q[4];
rz(62.65625) q[4];
cx q[8],q[5];
rz(2.34375) q[5];
cx q[8],q[3];
rz(-92.5625) q[3];
cx q[8],q[4];
rz(34.625) q[4];
cx q[8],q[3];
cx q[5],q[3];
rz(28.1875) q[3];
cx q[4],q[3];
rz(-93.5625) q[3];
cx q[8],q[3];
rz(-25.03125) q[3];
cx q[14],q[12];
cx q[12],q[11];
rz(2.34375) q[11];
cx q[12],q[9];
rz(-92.5625) q[9];
cx q[12],q[10];
rz(34.625) q[10];
cx q[13],q[11];
rz(44.65625) q[11];
cx q[13],q[9];
rz(20.875) q[9];
cx q[13],q[10];
rz(51.375) q[10];
cx q[11],q[9];
cx q[13],q[9];
rz(62.65625) q[9];
cx q[10],q[9];
rz(-25.03125) q[9];
cx q[13],q[9];
rz(-128.03125) q[9];
cx q[18],q[17];
cx q[19],q[17];
rz(19.3125) q[17];
cx q[20],q[17];
rz(44.65625) q[17];
cx q[17],q[15];
rz(-11.15625) q[15];
cx q[17],q[16];
rz(62.65625) q[16];
cx q[20],q[15];
rz(-92.5625) q[15];
cx q[20],q[16];
rz(34.625) q[16];
cx q[6],q[5];
cx q[5],q[1];
rz(-133.8125) q[1];
cx q[5],q[0];
rz(51.375) q[0];
cx q[5],q[4];
rz(24.84375) q[4];
cx q[5],q[3];
rz(20.875) q[3];
cx q[8],q[4];
rz(-11.125) q[4];
cx q[8],q[3];
rz(-126.65625) q[3];
cx q[5],q[0];
cx q[1],q[0];
rz(2.34375) q[0];
cx q[12],q[10];
cx q[11],q[10];
rz(-23.21875) q[10];
cx q[13],q[10];
rz(24.84375) q[10];
cx q[11],q[9];
cx q[12],q[9];
rz(-254.5625) q[9];
cx q[13],q[9];
rz(20.875) q[9];
cx q[21],q[19];
cx q[19],q[17];
rz(28.1875) q[17];
cx q[19],q[15];
rz(-25.03125) q[15];
cx q[19],q[16];
rz(83.71875) q[16];
cx q[20],q[17];
rz(2.34375) q[17];
cx q[20],q[15];
rz(20.875) q[15];
cx q[20],q[16];
rz(51.375) q[16];
cx q[21],q[2];
cx q[2],q[1];
rz(-63.125) q[1];
cx q[2],q[0];
rz(62.65625) q[0];
cx q[5],q[0];
rz(-11.125) q[0];
cx q[4],q[3];
cx q[6],q[3];
rz(-11.15625) q[3];
cx q[8],q[3];
rz(-126.125) q[3];
cx q[12],q[10];
cx q[10],q[9];
rz(-126.65625) q[9];
cx q[13],q[9];
rz(-11.15625) q[9];
cx q[19],q[15];
cx q[17],q[15];
rz(28.1875) q[15];
cx q[16],q[15];
rz(-163.625) q[15];
cx q[17],q[15];
rz(-23.21875) q[15];
cx q[20],q[15];
rz(51.375) q[15];
cx q[17],q[15];
rz(-128.03125) q[15];
cx q[19],q[15];
rz(-93.5625) q[15];
cx q[20],q[15];
rz(-25.03125) q[15];
cx q[2],q[0];
cx q[1],q[0];
rz(19.3125) q[0];
cx q[4],q[3];
cx q[3],q[1];
rz(-2.25) q[1];
cx q[3],q[0];
rz(-163.625) q[0];
cx q[2],q[1];
rz(34.625) q[1];
cx q[2],q[0];
rz(20.875) q[0];
cx q[5],q[1];
rz(62.65625) q[1];
cx q[5],q[0];
rz(-126.65625) q[0];
cx q[2],q[1];
rz(-23.21875) q[1];
cx q[2],q[0];
rz(-128.03125) q[0];
cx q[21],q[18];
cx q[18],q[15];
rz(-2.25) q[15];
cx q[20],q[15];
rz(83.71875) q[15];
cx q[17],q[15];
rz(-254.5625) q[15];
cx q[20],q[15];
rz(-128.03125) q[15];
cx q[19],q[15];
rz(20.875) q[15];
cx q[20],q[15];
rz(-126.65625) q[15];
cx q[16],q[15];
rz(24.84375) q[15];
cx q[20],q[15];
rz(34.625) q[15];
cx q[17],q[15];
rz(-126.65625) q[15];
cx q[20],q[15];
rz(-93.5625) q[15];
cx q[19],q[15];
rz(-11.15625) q[15];
cx q[20],q[15];
rz(-126.125) q[15];
cx q[3],q[0];
cx q[1],q[0];
rz(20.875) q[0];
cx q[2],q[0];
rz(-126.125) q[0];
cx q[5],q[0];
rz(-11.15625) q[0];
cx q[2],q[0];
rz(-25.03125) q[0];
cx q[18],q[16];
cx q[17],q[16];
rz(-11.125) q[16];
cx q[20],q[16];
rz(24.84375) q[16];
cx q[19],q[16];
rz(-2.25) q[16];
cx q[20],q[16];
rz(-23.21875) q[16];
cx q[17],q[16];
rz(-133.8125) q[16];
cx q[20],q[16];
rz(3.03125) q[16];
cx q[8],q[6];
cx q[6],q[4];
cx q[4],q[1];
rz(-11.125) q[1];
cx q[4],q[0];
rz(-92.5625) q[0];
cx q[2],q[1];
rz(51.375) q[1];
cx q[2],q[0];
rz(-93.5625) q[0];
cx q[5],q[1];
rz(83.71875) q[1];
cx q[5],q[0];
rz(-126.65625) q[0];
cx q[2],q[1];
rz(24.84375) q[1];
cx q[2],q[0];
rz(-11.15625) q[0];
cx q[3],q[1];
rz(3.03125) q[1];
cx q[3],q[0];
rz(24.84375) q[0];
cx q[2],q[1];
rz(-4.03125) q[1];
cx q[5],q[0];
rz(34.625) q[0];
cx q[2],q[0];
rz(28.1875) q[0];
cx q[5],q[2];
cx q[2],q[1];
rz(-102.75) q[1];
cx q[1],q[0];
cx q[4],q[0];
rz(44.65625) q[0];
cx q[2],q[0];
rz(-2.25) q[0];
cx q[5],q[0];
rz(83.71875) q[0];
cx q[3],q[0];
rz(-25.03125) q[0];
cx q[5],q[0];
rz(-93.5625) q[0];
cx q[2],q[0];
rz(-128.03125) q[0];
cx q[5],q[0];
rz(-254.5625) q[0];
cx q[9],q[7];
cx q[10],q[7];
cx q[11],q[7];
cx q[7],q[5];
rz(-63.125) q[5];
cx q[7],q[3];
rz(-23.21875) q[3];
cx q[7],q[4];
rz(-102.75) q[4];
cx q[8],q[5];
rz(-4.03125) q[5];
cx q[8],q[3];
rz(51.375) q[3];
cx q[6],q[4];
rz(-23.21875) q[4];
cx q[6],q[5];
rz(19.3125) q[5];
cx q[6],q[3];
rz(-128.03125) q[3];
cx q[8],q[4];
rz(-2.25) q[4];
cx q[8],q[5];
rz(44.65625) q[5];
cx q[8],q[3];
rz(-254.5625) q[3];
cx q[6],q[4];
rz(3.03125) q[4];
cx q[7],q[5];
cx q[5],q[3];
rz(-2.25) q[3];
cx q[5],q[4];
rz(83.71875) q[4];
cx q[8],q[3];
rz(83.71875) q[3];
cx q[6],q[4];
rz(-133.8125) q[4];
cx q[6],q[3];
rz(-163.625) q[3];
cx q[8],q[4];
rz(3.03125) q[4];
cx q[8],q[3];
rz(-128.03125) q[3];
cx q[6],q[4];
rz(51.375) q[4];
cx q[4],q[3];
cx q[7],q[3];
rz(-11.125) q[3];
cx q[6],q[3];
rz(-126.65625) q[3];
cx q[8],q[3];
rz(-93.5625) q[3];
cx q[6],q[3];
rz(62.65625) q[3];
cx q[5],q[3];
rz(-25.03125) q[3];
cx q[6],q[3];
rz(34.625) q[3];
cx q[8],q[3];
rz(24.84375) q[3];
cx q[6],q[3];
rz(20.875) q[3];
cx q[15],q[14];
cx q[16],q[14];
cx q[18],q[14];
cx q[20],q[14];
cx q[14],q[11];
rz(19.3125) q[11];
cx q[14],q[10];
rz(-102.75) q[10];
cx q[14],q[9];
rz(-126.125) q[9];
cx q[13],q[11];
rz(28.1875) q[11];
cx q[12],q[10];
rz(-11.125) q[10];
cx q[13],q[9];
rz(-93.5625) q[9];
cx q[12],q[11];
rz(-63.125) q[11];
cx q[13],q[10];
rz(-2.25) q[10];
cx q[12],q[9];
rz(28.1875) q[9];
cx q[13],q[11];
rz(-120.375) q[11];
cx q[12],q[10];
rz(-63.125) q[10];
cx q[13],q[9];
rz(-11.125) q[9];
cx q[14],q[11];
cx q[11],q[10];
rz(-4.03125) q[10];
cx q[11],q[9];
rz(24.84375) q[9];
cx q[12],q[10];
rz(83.71875) q[10];
cx q[13],q[9];
rz(2.34375) q[9];
cx q[13],q[10];
rz(62.65625) q[10];
cx q[12],q[9];
rz(-25.03125) q[9];
cx q[12],q[10];
rz(3.03125) q[10];
cx q[13],q[9];
rz(-11.15625) q[9];
cx q[10],q[9];
cx q[14],q[9];
rz(-128.03125) q[9];
cx q[13],q[9];
rz(-126.65625) q[9];
cx q[12],q[9];
rz(-23.21875) q[9];
cx q[13],q[9];
rz(44.65625) q[9];
cx q[11],q[9];
rz(19.3125) q[9];
cx q[13],q[9];
rz(-2.25) q[9];
cx q[12],q[9];
rz(-93.5625) q[9];
cx q[13],q[9];
rz(-163.625) q[9];
cx q[2],q[1];
cx q[6],q[5];
cx q[5],q[1];
cx q[1],q[0];
cx q[6],q[3];
cx q[7],q[3];
cx q[3],q[0];
cx q[6],q[4];
cx q[7],q[4];
cx q[4],q[2];
cx q[4],q[0];
cx q[8],q[7];
cx q[8],q[6];
cx q[13],q[10];
cx q[10],q[9];
cx q[11],q[9];
cx q[12],q[9];
cx q[13],q[12];
cx q[14],q[10];
cx q[17],q[15];
cx q[19],q[18];
cx q[18],q[17];
cx q[19],q[16];
cx q[20],q[19];
cx q[20],q[15];
cx q[22],q[21];
cx q[23],q[22];
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
