OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
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
cx q[0],q[5];
rz(6.24980210738301) q[5];
cx q[0],q[5];
cx q[0],q[6];
rz(7.487147473462113) q[6];
cx q[0],q[6];
cx q[10],q[0];
rz(6.927190393297324) q[0];
cx q[10],q[0];
cx q[1],q[6];
rz(5.447362228018133) q[6];
cx q[1],q[6];
cx q[1],q[7];
rz(6.2252382750986595) q[7];
cx q[1],q[7];
cx q[9],q[1];
rz(6.5116529132051815) q[1];
cx q[9],q[1];
cx q[2],q[4];
rz(6.10407348841046) q[4];
cx q[2],q[4];
cx q[2],q[8];
rz(5.310890048556416) q[8];
cx q[2],q[8];
cx q[9],q[2];
rz(6.926309887180551) q[2];
cx q[9],q[2];
cx q[3],q[7];
rz(6.78506287081974) q[7];
cx q[3],q[7];
cx q[3],q[10];
rz(5.575212660282937) q[10];
cx q[3],q[10];
cx q[11],q[3];
rz(5.886880178642986) q[3];
cx q[11],q[3];
cx q[4],q[9];
rz(5.487492339121077) q[9];
cx q[4],q[9];
cx q[10],q[4];
rz(7.111568022138512) q[4];
cx q[10],q[4];
cx q[5],q[8];
rz(5.751639468129063) q[8];
cx q[5],q[8];
cx q[11],q[5];
rz(6.610117354150527) q[5];
cx q[11],q[5];
cx q[11],q[6];
rz(6.3237517647238235) q[6];
cx q[11],q[6];
cx q[8],q[7];
rz(7.043551886342001) q[7];
cx q[8],q[7];
rx(3.0957883399316963) q[0];
rx(3.0957883399316963) q[1];
rx(3.0957883399316963) q[2];
rx(3.0957883399316963) q[3];
rx(3.0957883399316963) q[4];
rx(3.0957883399316963) q[5];
rx(3.0957883399316963) q[6];
rx(3.0957883399316963) q[7];
rx(3.0957883399316963) q[8];
rx(3.0957883399316963) q[9];
rx(3.0957883399316963) q[10];
rx(3.0957883399316963) q[11];
cx q[0],q[5];
rz(3.3833002619765704) q[5];
cx q[0],q[5];
cx q[0],q[6];
rz(4.053131214906354) q[6];
cx q[0],q[6];
cx q[10],q[0];
rz(3.7500011471912247) q[0];
cx q[10],q[0];
cx q[1],q[6];
rz(2.9489032990921817) q[6];
cx q[1],q[6];
cx q[1],q[7];
rz(3.3700027497074037) q[7];
cx q[1],q[7];
cx q[9],q[1];
rz(3.5250519342239808) q[1];
cx q[9],q[1];
cx q[2],q[4];
rz(3.3044107761535124) q[4];
cx q[2],q[4];
cx q[2],q[8];
rz(2.875024742204773) q[8];
cx q[2],q[8];
cx q[9],q[2];
rz(3.749524489446809) q[2];
cx q[9],q[2];
cx q[3],q[7];
rz(3.6730610976071842) q[7];
cx q[3],q[7];
cx q[3],q[10];
rz(3.0181145146704065) q[10];
cx q[3],q[10];
cx q[11],q[3];
rz(3.186834224254728) q[3];
cx q[11],q[3];
cx q[4],q[9];
rz(2.9706275414081658) q[9];
cx q[4],q[9];
cx q[10],q[4];
rz(3.849813088312384) q[4];
cx q[10],q[4];
cx q[5],q[8];
rz(3.1136223171495203) q[8];
cx q[5],q[8];
cx q[11],q[5];
rz(3.578355184970469) q[5];
cx q[11],q[5];
cx q[11],q[6];
rz(3.423332552720416) q[6];
cx q[11],q[6];
cx q[8],q[7];
rz(3.8129928808713953) q[7];
cx q[8],q[7];
rx(3.8255494988769976) q[0];
rx(3.8255494988769976) q[1];
rx(3.8255494988769976) q[2];
rx(3.8255494988769976) q[3];
rx(3.8255494988769976) q[4];
rx(3.8255494988769976) q[5];
rx(3.8255494988769976) q[6];
rx(3.8255494988769976) q[7];
rx(3.8255494988769976) q[8];
rx(3.8255494988769976) q[9];
rx(3.8255494988769976) q[10];
rx(3.8255494988769976) q[11];
