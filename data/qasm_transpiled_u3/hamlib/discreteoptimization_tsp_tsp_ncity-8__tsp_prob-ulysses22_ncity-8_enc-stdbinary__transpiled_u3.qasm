OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
u3(0,0,-154.625) q[0];
u3(0,0,271.625) q[1];
u3(0,0,272.3125) q[2];
u3(0,0,-154.625) q[3];
u3(0,0,271.625) q[4];
u3(0,0,272.3125) q[5];
cx q[5],q[4];
u3(0,0,311.4375) q[4];
cx q[5],q[3];
u3(0,0,-125.6875) q[3];
cx q[4],q[3];
u3(0,0,-85.625) q[3];
cx q[5],q[3];
u3(0,0,-264.3125) q[3];
u3(0,0,-154.625) q[6];
u3(0,0,271.625) q[7];
u3(0,0,272.3125) q[8];
u3(0,0,-154.625) q[9];
u3(0,0,271.625) q[10];
u3(0,0,272.3125) q[11];
cx q[11],q[10];
u3(0,0,311.4375) q[10];
cx q[11],q[9];
u3(0,0,-125.6875) q[9];
cx q[10],q[9];
u3(0,0,-85.625) q[9];
cx q[11],q[8];
u3(0,0,-120.375) q[8];
cx q[10],q[8];
u3(0,0,-4.03125) q[8];
cx q[11],q[7];
u3(0,0,-4.03125) q[7];
cx q[10],q[7];
u3(0,0,-133.8125) q[7];
cx q[11],q[6];
u3(0,0,2.34375) q[6];
cx q[10],q[6];
u3(0,0,34.625) q[6];
cx q[11],q[9];
u3(0,0,-264.3125) q[9];
cx q[11],q[8];
u3(0,0,-63.125) q[8];
cx q[11],q[7];
u3(0,0,3.03125) q[7];
cx q[11],q[6];
u3(0,0,24.84375) q[6];
cx q[9],q[8];
u3(0,0,2.34375) q[8];
cx q[11],q[8];
u3(0,0,28.1875) q[8];
cx q[10],q[8];
u3(0,0,44.65625) q[8];
cx q[9],q[7];
u3(0,0,34.625) q[7];
cx q[11],q[7];
u3(0,0,62.65625) q[7];
cx q[10],q[7];
u3(0,0,51.375) q[7];
cx q[9],q[6];
u3(0,0,-92.5625) q[6];
cx q[11],q[6];
u3(0,0,-11.15625) q[6];
cx q[10],q[6];
u3(0,0,20.875) q[6];
cx q[11],q[8];
u3(0,0,19.3125) q[8];
cx q[11],q[7];
u3(0,0,83.71875) q[7];
cx q[11],q[6];
u3(0,0,-25.03125) q[6];
cx q[8],q[7];
u3(0,0,311.4375) q[7];
cx q[11],q[7];
u3(0,0,-63.125) q[7];
cx q[10],q[7];
u3(0,0,3.03125) q[7];
cx q[8],q[6];
u3(0,0,-125.6875) q[6];
cx q[11],q[6];
u3(0,0,28.1875) q[6];
cx q[10],q[6];
u3(0,0,62.65625) q[6];
cx q[11],q[7];
u3(0,0,-102.75) q[7];
cx q[11],q[6];
u3(0,0,-11.125) q[6];
cx q[9],q[7];
u3(0,0,24.84375) q[7];
cx q[11],q[7];
u3(0,0,-11.125) q[7];
cx q[10],q[7];
u3(0,0,-23.21875) q[7];
cx q[9],q[6];
u3(0,0,-11.15625) q[6];
cx q[11],q[6];
u3(0,0,-126.125) q[6];
cx q[10],q[6];
u3(0,0,-126.65625) q[6];
cx q[11],q[7];
u3(0,0,-2.25) q[7];
cx q[11],q[6];
u3(0,0,-93.5625) q[6];
cx q[7],q[6];
u3(0,0,-85.625) q[6];
cx q[11],q[6];
u3(0,0,44.65625) q[6];
cx q[10],q[6];
u3(0,0,51.375) q[6];
cx q[11],q[6];
u3(0,0,-23.21875) q[6];
cx q[9],q[6];
u3(0,0,20.875) q[6];
cx q[11],q[6];
u3(0,0,-126.65625) q[6];
cx q[10],q[6];
u3(0,0,-254.5625) q[6];
cx q[11],q[6];
u3(0,0,-128.03125) q[6];
cx q[8],q[6];
u3(0,0,-264.3125) q[6];
cx q[11],q[6];
u3(0,0,19.3125) q[6];
cx q[10],q[6];
u3(0,0,83.71875) q[6];
cx q[11],q[6];
u3(0,0,-2.25) q[6];
cx q[9],q[6];
u3(0,0,-25.03125) q[6];
cx q[11],q[6];
u3(0,0,-93.5625) q[6];
cx q[10],q[6];
u3(0,0,-128.03125) q[6];
cx q[11],q[6];
u3(0,0,-163.625) q[6];
cx q[7],q[6];
cx q[9],q[8];
u3(0,0,-154.625) q[12];
u3(0,0,271.625) q[13];
u3(0,0,272.3125) q[14];
u3(0,0,-154.625) q[15];
u3(0,0,271.625) q[16];
u3(0,0,272.3125) q[17];
cx q[17],q[16];
u3(0,0,311.4375) q[16];
cx q[17],q[15];
u3(0,0,-125.6875) q[15];
cx q[16],q[15];
u3(0,0,-85.625) q[15];
cx q[17],q[14];
u3(0,0,-120.375) q[14];
cx q[16],q[14];
u3(0,0,-4.03125) q[14];
cx q[17],q[13];
u3(0,0,-4.03125) q[13];
cx q[16],q[13];
u3(0,0,-133.8125) q[13];
cx q[17],q[12];
u3(0,0,2.34375) q[12];
cx q[16],q[12];
u3(0,0,34.625) q[12];
cx q[17],q[15];
u3(0,0,-264.3125) q[15];
cx q[17],q[14];
u3(0,0,-63.125) q[14];
cx q[15],q[14];
u3(0,0,2.34375) q[14];
cx q[17],q[13];
u3(0,0,3.03125) q[13];
cx q[15],q[13];
u3(0,0,34.625) q[13];
cx q[17],q[12];
u3(0,0,24.84375) q[12];
cx q[15],q[12];
u3(0,0,-92.5625) q[12];
cx q[17],q[14];
u3(0,0,28.1875) q[14];
cx q[16],q[14];
u3(0,0,44.65625) q[14];
cx q[17],q[13];
u3(0,0,62.65625) q[13];
cx q[16],q[13];
u3(0,0,51.375) q[13];
cx q[17],q[12];
u3(0,0,-11.15625) q[12];
cx q[16],q[12];
u3(0,0,20.875) q[12];
cx q[17],q[14];
u3(0,0,19.3125) q[14];
cx q[17],q[13];
u3(0,0,83.71875) q[13];
cx q[14],q[13];
u3(0,0,311.4375) q[13];
cx q[17],q[12];
u3(0,0,-25.03125) q[12];
cx q[14],q[12];
u3(0,0,-125.6875) q[12];
cx q[17],q[13];
u3(0,0,-63.125) q[13];
cx q[16],q[13];
u3(0,0,3.03125) q[13];
cx q[17],q[12];
u3(0,0,28.1875) q[12];
cx q[16],q[12];
u3(0,0,62.65625) q[12];
cx q[17],q[13];
u3(0,0,-102.75) q[13];
cx q[15],q[13];
u3(0,0,24.84375) q[13];
cx q[17],q[12];
u3(0,0,-11.125) q[12];
cx q[15],q[12];
u3(0,0,-11.15625) q[12];
cx q[17],q[13];
u3(0,0,-11.125) q[13];
cx q[16],q[13];
u3(0,0,-23.21875) q[13];
cx q[17],q[12];
u3(0,0,-126.125) q[12];
cx q[16],q[12];
u3(0,0,-126.65625) q[12];
cx q[17],q[13];
u3(0,0,-2.25) q[13];
cx q[17],q[12];
u3(0,0,-93.5625) q[12];
cx q[13],q[12];
u3(0,0,-85.625) q[12];
cx q[17],q[12];
u3(0,0,44.65625) q[12];
cx q[16],q[12];
u3(0,0,51.375) q[12];
cx q[17],q[12];
u3(0,0,-23.21875) q[12];
cx q[15],q[12];
u3(0,0,20.875) q[12];
cx q[17],q[12];
u3(0,0,-126.65625) q[12];
cx q[16],q[12];
u3(0,0,-254.5625) q[12];
cx q[17],q[12];
u3(0,0,-128.03125) q[12];
cx q[14],q[12];
u3(0,0,-264.3125) q[12];
cx q[14],q[13];
cx q[13],q[11];
u3(0,0,-4.03125) q[11];
cx q[13],q[10];
u3(0,0,3.03125) q[10];
cx q[13],q[9];
u3(0,0,83.71875) q[9];
cx q[13],q[9];
cx q[11],q[9];
u3(0,0,51.375) q[9];
cx q[11],q[10];
cx q[10],q[9];
u3(0,0,34.625) q[9];
cx q[13],q[10];
u3(0,0,-133.8125) q[10];
cx q[17],q[12];
u3(0,0,19.3125) q[12];
cx q[16],q[12];
u3(0,0,83.71875) q[12];
cx q[17],q[12];
u3(0,0,-2.25) q[12];
cx q[15],q[12];
u3(0,0,-25.03125) q[12];
cx q[17],q[12];
u3(0,0,-93.5625) q[12];
cx q[16],q[12];
u3(0,0,-128.03125) q[12];
cx q[17],q[12];
u3(0,0,-163.625) q[12];
cx q[14],q[12];
cx q[12],q[11];
u3(0,0,2.34375) q[11];
cx q[12],q[9];
u3(0,0,-92.5625) q[9];
cx q[12],q[10];
u3(0,0,34.625) q[10];
cx q[13],q[11];
u3(0,0,44.65625) q[11];
cx q[13],q[9];
u3(0,0,20.875) q[9];
cx q[11],q[9];
cx q[13],q[10];
u3(0,0,51.375) q[10];
cx q[13],q[9];
u3(0,0,62.65625) q[9];
cx q[10],q[9];
u3(0,0,-25.03125) q[9];
cx q[12],q[10];
cx q[11],q[10];
u3(0,0,-23.21875) q[10];
cx q[13],q[9];
u3(0,0,-128.03125) q[9];
cx q[11],q[9];
cx q[12],q[9];
u3(0,0,-254.5625) q[9];
cx q[13],q[10];
u3(0,0,24.84375) q[10];
cx q[12],q[10];
cx q[13],q[9];
u3(0,0,20.875) q[9];
cx q[10],q[9];
u3(0,0,-126.65625) q[9];
cx q[13],q[9];
u3(0,0,-11.15625) q[9];
cx q[9],q[7];
cx q[10],q[7];
cx q[11],q[7];
u3(0,0,-154.625) q[18];
u3(0,0,271.625) q[19];
u3(0,0,272.3125) q[20];
u3(0,0,-154.625) q[21];
u3(0,0,271.625) q[22];
u3(0,0,272.3125) q[23];
cx q[23],q[22];
u3(0,0,311.4375) q[22];
cx q[23],q[21];
u3(0,0,-125.6875) q[21];
cx q[22],q[21];
u3(0,0,-85.625) q[21];
cx q[23],q[20];
u3(0,0,-120.375) q[20];
cx q[22],q[20];
u3(0,0,-4.03125) q[20];
cx q[23],q[19];
u3(0,0,-4.03125) q[19];
cx q[22],q[19];
u3(0,0,-133.8125) q[19];
cx q[23],q[18];
u3(0,0,2.34375) q[18];
cx q[22],q[18];
u3(0,0,34.625) q[18];
cx q[23],q[2];
u3(0,0,-120.375) q[2];
cx q[22],q[2];
u3(0,0,-4.03125) q[2];
cx q[23],q[1];
u3(0,0,-4.03125) q[1];
cx q[22],q[1];
u3(0,0,-133.8125) q[1];
cx q[23],q[0];
u3(0,0,2.34375) q[0];
cx q[22],q[0];
u3(0,0,34.625) q[0];
cx q[23],q[21];
u3(0,0,-264.3125) q[21];
cx q[23],q[20];
u3(0,0,-63.125) q[20];
cx q[21],q[20];
u3(0,0,2.34375) q[20];
cx q[23],q[19];
u3(0,0,3.03125) q[19];
cx q[21],q[19];
u3(0,0,34.625) q[19];
cx q[23],q[18];
u3(0,0,24.84375) q[18];
cx q[21],q[18];
u3(0,0,-92.5625) q[18];
cx q[23],q[2];
u3(0,0,-63.125) q[2];
cx q[21],q[2];
u3(0,0,2.34375) q[2];
cx q[23],q[1];
u3(0,0,3.03125) q[1];
cx q[21],q[1];
u3(0,0,34.625) q[1];
cx q[23],q[0];
u3(0,0,24.84375) q[0];
cx q[21],q[0];
u3(0,0,-92.5625) q[0];
cx q[23],q[20];
u3(0,0,28.1875) q[20];
cx q[22],q[20];
u3(0,0,44.65625) q[20];
cx q[23],q[19];
u3(0,0,62.65625) q[19];
cx q[22],q[19];
u3(0,0,51.375) q[19];
cx q[23],q[18];
u3(0,0,-11.15625) q[18];
cx q[22],q[18];
u3(0,0,20.875) q[18];
cx q[23],q[2];
u3(0,0,28.1875) q[2];
cx q[22],q[2];
u3(0,0,44.65625) q[2];
cx q[23],q[1];
u3(0,0,62.65625) q[1];
cx q[22],q[1];
u3(0,0,51.375) q[1];
cx q[23],q[0];
u3(0,0,-11.15625) q[0];
cx q[22],q[0];
u3(0,0,20.875) q[0];
cx q[23],q[20];
u3(0,0,19.3125) q[20];
cx q[23],q[19];
u3(0,0,83.71875) q[19];
cx q[20],q[19];
u3(0,0,311.4375) q[19];
cx q[23],q[18];
u3(0,0,-25.03125) q[18];
cx q[20],q[18];
u3(0,0,-125.6875) q[18];
cx q[23],q[2];
u3(0,0,19.3125) q[2];
cx q[23],q[1];
u3(0,0,83.71875) q[1];
cx q[2],q[1];
u3(0,0,311.4375) q[1];
cx q[23],q[0];
u3(0,0,-25.03125) q[0];
cx q[2],q[0];
u3(0,0,-125.6875) q[0];
cx q[23],q[19];
u3(0,0,-63.125) q[19];
cx q[22],q[19];
u3(0,0,3.03125) q[19];
cx q[23],q[18];
u3(0,0,28.1875) q[18];
cx q[22],q[18];
u3(0,0,62.65625) q[18];
cx q[23],q[1];
u3(0,0,-63.125) q[1];
cx q[22],q[1];
u3(0,0,3.03125) q[1];
cx q[23],q[0];
u3(0,0,28.1875) q[0];
cx q[22],q[0];
u3(0,0,62.65625) q[0];
cx q[23],q[19];
u3(0,0,-102.75) q[19];
cx q[21],q[19];
u3(0,0,24.84375) q[19];
cx q[23],q[18];
u3(0,0,-11.125) q[18];
cx q[21],q[18];
u3(0,0,-11.15625) q[18];
cx q[23],q[1];
u3(0,0,-102.75) q[1];
cx q[21],q[1];
u3(0,0,24.84375) q[1];
cx q[23],q[0];
u3(0,0,-11.125) q[0];
cx q[21],q[0];
u3(0,0,-11.15625) q[0];
cx q[23],q[19];
u3(0,0,-11.125) q[19];
cx q[22],q[19];
u3(0,0,-23.21875) q[19];
cx q[23],q[18];
u3(0,0,-126.125) q[18];
cx q[22],q[18];
u3(0,0,-126.65625) q[18];
cx q[23],q[1];
u3(0,0,-11.125) q[1];
cx q[22],q[1];
u3(0,0,-23.21875) q[1];
cx q[23],q[0];
u3(0,0,-126.125) q[0];
cx q[22],q[0];
u3(0,0,-126.65625) q[0];
cx q[23],q[19];
u3(0,0,-2.25) q[19];
cx q[23],q[18];
u3(0,0,-93.5625) q[18];
cx q[19],q[18];
u3(0,0,-85.625) q[18];
cx q[23],q[1];
u3(0,0,-2.25) q[1];
cx q[23],q[0];
u3(0,0,-93.5625) q[0];
cx q[1],q[0];
u3(0,0,-85.625) q[0];
cx q[23],q[18];
u3(0,0,44.65625) q[18];
cx q[22],q[18];
u3(0,0,51.375) q[18];
cx q[23],q[0];
u3(0,0,44.65625) q[0];
cx q[22],q[0];
u3(0,0,51.375) q[0];
cx q[23],q[18];
u3(0,0,-23.21875) q[18];
cx q[21],q[18];
u3(0,0,20.875) q[18];
cx q[23],q[0];
u3(0,0,-23.21875) q[0];
cx q[21],q[0];
u3(0,0,20.875) q[0];
cx q[23],q[18];
u3(0,0,-126.65625) q[18];
cx q[22],q[18];
u3(0,0,-254.5625) q[18];
cx q[23],q[0];
u3(0,0,-126.65625) q[0];
cx q[22],q[0];
u3(0,0,-254.5625) q[0];
cx q[23],q[18];
u3(0,0,-128.03125) q[18];
cx q[20],q[18];
u3(0,0,-264.3125) q[18];
cx q[23],q[0];
u3(0,0,-128.03125) q[0];
cx q[2],q[0];
u3(0,0,-264.3125) q[0];
cx q[23],q[18];
u3(0,0,19.3125) q[18];
cx q[22],q[18];
u3(0,0,83.71875) q[18];
cx q[23],q[0];
u3(0,0,19.3125) q[0];
cx q[22],q[0];
u3(0,0,83.71875) q[0];
cx q[23],q[18];
u3(0,0,-2.25) q[18];
cx q[21],q[18];
u3(0,0,-25.03125) q[18];
cx q[23],q[0];
u3(0,0,-2.25) q[0];
cx q[21],q[0];
u3(0,0,-25.03125) q[0];
cx q[21],q[20];
cx q[20],q[17];
u3(0,0,-120.375) q[17];
cx q[20],q[16];
u3(0,0,-63.125) q[16];
cx q[20],q[15];
u3(0,0,19.3125) q[15];
cx q[20],q[15];
cx q[17],q[15];
u3(0,0,44.65625) q[15];
cx q[17],q[16];
cx q[16],q[15];
u3(0,0,2.34375) q[15];
cx q[20],q[16];
u3(0,0,-4.03125) q[16];
cx q[21],q[2];
cx q[21],q[17];
cx q[19],q[17];
u3(0,0,-4.03125) q[17];
cx q[20],q[17];
u3(0,0,-63.125) q[17];
cx q[17],q[16];
u3(0,0,3.03125) q[16];
cx q[17],q[15];
u3(0,0,62.65625) q[15];
cx q[20],q[16];
u3(0,0,-102.75) q[16];
cx q[20],q[15];
u3(0,0,-11.125) q[15];
cx q[23],q[18];
u3(0,0,-93.5625) q[18];
cx q[22],q[18];
u3(0,0,-128.03125) q[18];
cx q[23],q[0];
u3(0,0,-93.5625) q[0];
cx q[22],q[0];
u3(0,0,-128.03125) q[0];
cx q[23],q[18];
u3(0,0,-163.625) q[18];
cx q[18],q[17];
cx q[19],q[17];
u3(0,0,19.3125) q[17];
cx q[20],q[17];
u3(0,0,44.65625) q[17];
cx q[17],q[15];
u3(0,0,-11.15625) q[15];
cx q[17],q[16];
u3(0,0,62.65625) q[16];
cx q[20],q[15];
u3(0,0,-92.5625) q[15];
cx q[20],q[16];
u3(0,0,34.625) q[16];
cx q[23],q[0];
u3(0,0,-163.625) q[0];
cx q[5],q[2];
u3(0,0,-120.375) q[2];
cx q[3],q[2];
u3(0,0,44.65625) q[2];
cx q[5],q[2];
u3(0,0,19.3125) q[2];
cx q[4],q[2];
u3(0,0,2.34375) q[2];
cx q[5],q[2];
u3(0,0,28.1875) q[2];
cx q[3],q[2];
u3(0,0,-4.03125) q[2];
cx q[5],q[2];
u3(0,0,-63.125) q[2];
cx q[21],q[2];
cx q[2],q[1];
u3(0,0,3.03125) q[1];
cx q[2],q[0];
u3(0,0,-23.21875) q[0];
cx q[21],q[19];
cx q[19],q[17];
u3(0,0,28.1875) q[17];
cx q[19],q[15];
u3(0,0,-25.03125) q[15];
cx q[19],q[16];
u3(0,0,83.71875) q[16];
cx q[20],q[17];
u3(0,0,2.34375) q[17];
cx q[20],q[15];
u3(0,0,20.875) q[15];
cx q[19],q[15];
cx q[17],q[15];
u3(0,0,28.1875) q[15];
cx q[20],q[16];
u3(0,0,51.375) q[16];
cx q[16],q[15];
u3(0,0,-163.625) q[15];
cx q[17],q[15];
u3(0,0,-23.21875) q[15];
cx q[20],q[15];
u3(0,0,51.375) q[15];
cx q[17],q[15];
u3(0,0,-128.03125) q[15];
cx q[19],q[15];
u3(0,0,-93.5625) q[15];
cx q[20],q[15];
u3(0,0,-25.03125) q[15];
cx q[21],q[2];
cx q[21],q[18];
cx q[18],q[15];
u3(0,0,-2.25) q[15];
cx q[20],q[15];
u3(0,0,83.71875) q[15];
cx q[17],q[15];
u3(0,0,-254.5625) q[15];
cx q[20],q[15];
u3(0,0,-128.03125) q[15];
cx q[19],q[15];
u3(0,0,20.875) q[15];
cx q[20],q[15];
u3(0,0,-126.65625) q[15];
cx q[16],q[15];
u3(0,0,24.84375) q[15];
cx q[18],q[16];
cx q[20],q[15];
u3(0,0,34.625) q[15];
cx q[17],q[15];
u3(0,0,-126.65625) q[15];
cx q[17],q[16];
u3(0,0,-11.125) q[16];
cx q[20],q[15];
u3(0,0,-93.5625) q[15];
cx q[19],q[15];
u3(0,0,-11.15625) q[15];
cx q[20],q[15];
u3(0,0,-126.125) q[15];
cx q[15],q[14];
cx q[20],q[16];
u3(0,0,24.84375) q[16];
cx q[19],q[16];
u3(0,0,-2.25) q[16];
cx q[20],q[16];
u3(0,0,-23.21875) q[16];
cx q[17],q[16];
u3(0,0,-133.8125) q[16];
cx q[17],q[15];
cx q[20],q[16];
u3(0,0,3.03125) q[16];
cx q[16],q[14];
cx q[18],q[14];
cx q[19],q[18];
cx q[18],q[17];
cx q[19],q[16];
cx q[20],q[14];
cx q[14],q[11];
u3(0,0,19.3125) q[11];
cx q[13],q[11];
u3(0,0,28.1875) q[11];
cx q[14],q[10];
u3(0,0,-102.75) q[10];
cx q[12],q[10];
u3(0,0,-11.125) q[10];
cx q[12],q[11];
u3(0,0,-63.125) q[11];
cx q[14],q[9];
u3(0,0,-126.125) q[9];
cx q[13],q[9];
u3(0,0,-93.5625) q[9];
cx q[12],q[9];
u3(0,0,28.1875) q[9];
cx q[13],q[10];
u3(0,0,-2.25) q[10];
cx q[12],q[10];
u3(0,0,-63.125) q[10];
cx q[13],q[11];
u3(0,0,-120.375) q[11];
cx q[13],q[9];
u3(0,0,-11.125) q[9];
cx q[14],q[11];
cx q[11],q[10];
u3(0,0,-4.03125) q[10];
cx q[11],q[9];
u3(0,0,24.84375) q[9];
cx q[12],q[10];
u3(0,0,83.71875) q[10];
cx q[13],q[9];
u3(0,0,2.34375) q[9];
cx q[12],q[9];
u3(0,0,-25.03125) q[9];
cx q[13],q[10];
u3(0,0,62.65625) q[10];
cx q[12],q[10];
u3(0,0,3.03125) q[10];
cx q[13],q[9];
u3(0,0,-11.15625) q[9];
cx q[10],q[9];
cx q[14],q[9];
u3(0,0,-128.03125) q[9];
cx q[13],q[9];
u3(0,0,-126.65625) q[9];
cx q[12],q[9];
u3(0,0,-23.21875) q[9];
cx q[13],q[9];
u3(0,0,44.65625) q[9];
cx q[11],q[9];
u3(0,0,19.3125) q[9];
cx q[13],q[9];
u3(0,0,-2.25) q[9];
cx q[12],q[9];
u3(0,0,-93.5625) q[9];
cx q[13],q[9];
u3(0,0,-163.625) q[9];
cx q[13],q[10];
cx q[10],q[9];
cx q[11],q[9];
cx q[12],q[9];
cx q[13],q[12];
cx q[14],q[10];
cx q[20],q[19];
cx q[20],q[15];
cx q[22],q[21];
cx q[23],q[22];
cx q[8],q[5];
u3(0,0,-120.375) q[5];
cx q[8],q[4];
u3(0,0,-63.125) q[4];
cx q[8],q[3];
u3(0,0,19.3125) q[3];
cx q[8],q[3];
cx q[5],q[3];
u3(0,0,44.65625) q[3];
cx q[5],q[4];
cx q[4],q[3];
u3(0,0,2.34375) q[3];
cx q[6],q[5];
u3(0,0,28.1875) q[5];
cx q[6],q[3];
u3(0,0,-11.15625) q[3];
cx q[8],q[4];
u3(0,0,-4.03125) q[4];
cx q[6],q[4];
u3(0,0,62.65625) q[4];
cx q[8],q[5];
u3(0,0,2.34375) q[5];
cx q[8],q[3];
u3(0,0,-92.5625) q[3];
cx q[8],q[4];
u3(0,0,34.625) q[4];
cx q[8],q[3];
cx q[5],q[3];
u3(0,0,28.1875) q[3];
cx q[4],q[3];
u3(0,0,-93.5625) q[3];
cx q[6],q[5];
cx q[5],q[1];
u3(0,0,-133.8125) q[1];
cx q[5],q[0];
u3(0,0,51.375) q[0];
cx q[5],q[4];
u3(0,0,24.84375) q[4];
cx q[8],q[3];
u3(0,0,-25.03125) q[3];
cx q[5],q[3];
u3(0,0,20.875) q[3];
cx q[5],q[0];
cx q[1],q[0];
u3(0,0,2.34375) q[0];
cx q[2],q[1];
u3(0,0,-63.125) q[1];
cx q[2],q[0];
u3(0,0,62.65625) q[0];
cx q[5],q[0];
u3(0,0,-11.125) q[0];
cx q[2],q[0];
cx q[1],q[0];
u3(0,0,19.3125) q[0];
cx q[8],q[4];
u3(0,0,-11.125) q[4];
cx q[8],q[3];
u3(0,0,-126.65625) q[3];
cx q[4],q[3];
cx q[6],q[3];
u3(0,0,-11.15625) q[3];
cx q[8],q[3];
u3(0,0,-126.125) q[3];
cx q[4],q[3];
cx q[3],q[1];
u3(0,0,-2.25) q[1];
cx q[2],q[1];
u3(0,0,34.625) q[1];
cx q[3],q[0];
u3(0,0,-163.625) q[0];
cx q[2],q[0];
u3(0,0,20.875) q[0];
cx q[5],q[1];
u3(0,0,62.65625) q[1];
cx q[2],q[1];
u3(0,0,-23.21875) q[1];
cx q[5],q[0];
u3(0,0,-126.65625) q[0];
cx q[2],q[0];
u3(0,0,-128.03125) q[0];
cx q[3],q[0];
cx q[1],q[0];
u3(0,0,20.875) q[0];
cx q[2],q[0];
u3(0,0,-126.125) q[0];
cx q[5],q[0];
u3(0,0,-11.15625) q[0];
cx q[2],q[0];
u3(0,0,-25.03125) q[0];
cx q[8],q[6];
cx q[6],q[4];
cx q[4],q[1];
u3(0,0,-11.125) q[1];
cx q[2],q[1];
u3(0,0,51.375) q[1];
cx q[4],q[0];
u3(0,0,-92.5625) q[0];
cx q[2],q[0];
u3(0,0,-93.5625) q[0];
cx q[5],q[1];
u3(0,0,83.71875) q[1];
cx q[2],q[1];
u3(0,0,24.84375) q[1];
cx q[3],q[1];
u3(0,0,3.03125) q[1];
cx q[5],q[0];
u3(0,0,-126.65625) q[0];
cx q[2],q[0];
u3(0,0,-11.15625) q[0];
cx q[2],q[1];
u3(0,0,-4.03125) q[1];
cx q[3],q[0];
u3(0,0,24.84375) q[0];
cx q[5],q[0];
u3(0,0,34.625) q[0];
cx q[2],q[0];
u3(0,0,28.1875) q[0];
cx q[5],q[2];
cx q[2],q[1];
u3(0,0,-102.75) q[1];
cx q[1],q[0];
cx q[4],q[0];
u3(0,0,44.65625) q[0];
cx q[2],q[0];
u3(0,0,-2.25) q[0];
cx q[5],q[0];
u3(0,0,83.71875) q[0];
cx q[3],q[0];
u3(0,0,-25.03125) q[0];
cx q[5],q[0];
u3(0,0,-93.5625) q[0];
cx q[2],q[0];
u3(0,0,-128.03125) q[0];
cx q[2],q[1];
cx q[5],q[0];
u3(0,0,-254.5625) q[0];
cx q[7],q[5];
u3(0,0,-63.125) q[5];
cx q[7],q[3];
u3(0,0,-23.21875) q[3];
cx q[7],q[4];
u3(0,0,-102.75) q[4];
cx q[6],q[4];
u3(0,0,-23.21875) q[4];
cx q[8],q[5];
u3(0,0,-4.03125) q[5];
cx q[6],q[5];
u3(0,0,19.3125) q[5];
cx q[8],q[3];
u3(0,0,51.375) q[3];
cx q[6],q[3];
u3(0,0,-128.03125) q[3];
cx q[8],q[4];
u3(0,0,-2.25) q[4];
cx q[6],q[4];
u3(0,0,3.03125) q[4];
cx q[8],q[5];
u3(0,0,44.65625) q[5];
cx q[7],q[5];
cx q[8],q[3];
u3(0,0,-254.5625) q[3];
cx q[5],q[3];
u3(0,0,-2.25) q[3];
cx q[5],q[4];
u3(0,0,83.71875) q[4];
cx q[6],q[4];
u3(0,0,-133.8125) q[4];
cx q[8],q[3];
u3(0,0,83.71875) q[3];
cx q[6],q[3];
u3(0,0,-163.625) q[3];
cx q[8],q[4];
u3(0,0,3.03125) q[4];
cx q[6],q[4];
u3(0,0,51.375) q[4];
cx q[8],q[3];
u3(0,0,-128.03125) q[3];
cx q[4],q[3];
cx q[7],q[3];
u3(0,0,-11.125) q[3];
cx q[6],q[3];
u3(0,0,-126.65625) q[3];
cx q[8],q[3];
u3(0,0,-93.5625) q[3];
cx q[6],q[3];
u3(0,0,62.65625) q[3];
cx q[5],q[3];
u3(0,0,-25.03125) q[3];
cx q[6],q[3];
u3(0,0,34.625) q[3];
cx q[8],q[3];
u3(0,0,24.84375) q[3];
cx q[6],q[3];
u3(0,0,20.875) q[3];
cx q[6],q[5];
cx q[5],q[1];
cx q[1],q[0];
cx q[6],q[3];
cx q[6],q[4];
cx q[7],q[3];
cx q[3],q[0];
cx q[7],q[4];
cx q[4],q[2];
cx q[4],q[0];
cx q[8],q[7];
cx q[8],q[6];
