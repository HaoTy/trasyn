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
cx q[0],q[3];
rz(1.919706022491257) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(2.031087240323598) q[4];
cx q[0],q[4];
cx q[6],q[0];
rz(2.108523522800362) q[0];
cx q[6],q[0];
cx q[1],q[7];
rz(2.3233006932535902) q[7];
cx q[1],q[7];
cx q[1],q[12];
rz(1.8994742856275966) q[12];
cx q[1],q[12];
cx q[14],q[1];
rz(2.357273868508989) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(1.7686505867339712) q[4];
cx q[2],q[4];
cx q[2],q[6];
rz(1.5655294828331487) q[6];
cx q[2],q[6];
cx q[12],q[2];
rz(1.9588699237506713) q[2];
cx q[12],q[2];
cx q[3],q[7];
rz(2.065685795804407) q[7];
cx q[3],q[7];
cx q[17],q[3];
rz(1.9646996818995643) q[3];
cx q[17],q[3];
cx q[13],q[4];
rz(2.4464480489486715) q[4];
cx q[13],q[4];
cx q[5],q[9];
rz(2.619278959745617) q[9];
cx q[5],q[9];
cx q[5],q[11];
rz(2.2529137384322278) q[11];
cx q[5],q[11];
cx q[17],q[5];
rz(2.220713742681935) q[5];
cx q[17],q[5];
cx q[13],q[6];
rz(2.087330676190051) q[6];
cx q[13],q[6];
cx q[18],q[7];
rz(2.315140473714135) q[7];
cx q[18],q[7];
cx q[8],q[10];
rz(2.34558017041458) q[10];
cx q[8],q[10];
cx q[8],q[11];
rz(2.3761197058660155) q[11];
cx q[8],q[11];
cx q[15],q[8];
rz(2.236939109899384) q[8];
cx q[15],q[8];
cx q[9],q[10];
rz(2.2508005636910595) q[10];
cx q[9],q[10];
cx q[19],q[9];
rz(2.601226358232663) q[9];
cx q[19],q[9];
cx q[15],q[10];
rz(1.8230079363409872) q[10];
cx q[15],q[10];
cx q[19],q[11];
rz(2.1992534576253675) q[11];
cx q[19],q[11];
cx q[16],q[12];
rz(1.8744165480634214) q[12];
cx q[16],q[12];
cx q[14],q[13];
rz(2.4564800058930207) q[13];
cx q[14],q[13];
cx q[16],q[14];
rz(2.101731309563592) q[14];
cx q[16],q[14];
cx q[18],q[15];
rz(2.1317303660172624) q[15];
cx q[18],q[15];
cx q[17],q[16];
rz(2.0612015779182027) q[16];
cx q[17],q[16];
cx q[19],q[18];
rz(2.1653245231366482) q[18];
cx q[19],q[18];
rx(1.129786927133171) q[0];
rx(1.129786927133171) q[1];
rx(1.129786927133171) q[2];
rx(1.129786927133171) q[3];
rx(1.129786927133171) q[4];
rx(1.129786927133171) q[5];
rx(1.129786927133171) q[6];
rx(1.129786927133171) q[7];
rx(1.129786927133171) q[8];
rx(1.129786927133171) q[9];
rx(1.129786927133171) q[10];
rx(1.129786927133171) q[11];
rx(1.129786927133171) q[12];
rx(1.129786927133171) q[13];
rx(1.129786927133171) q[14];
rx(1.129786927133171) q[15];
rx(1.129786927133171) q[16];
rx(1.129786927133171) q[17];
rx(1.129786927133171) q[18];
rx(1.129786927133171) q[19];
cx q[0],q[3];
rz(2.5757663628968412) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(2.72521215875819) q[4];
cx q[0],q[4];
cx q[6],q[0];
rz(2.8291123233326516) q[0];
cx q[6],q[0];
cx q[1],q[7];
rz(3.117289682100148) q[7];
cx q[1],q[7];
cx q[1],q[12];
rz(2.548620421452761) q[12];
cx q[1],q[12];
cx q[14],q[1];
rz(3.162873204284495) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(2.3730876684520448) q[4];
cx q[2],q[4];
cx q[2],q[6];
rz(2.10054984188251) q[6];
cx q[2],q[6];
cx q[12],q[2];
rz(2.62831454387973) q[2];
cx q[12],q[2];
cx q[3],q[7];
rz(2.771634785122948) q[7];
cx q[3],q[7];
cx q[17],q[3];
rz(2.6361366243273685) q[3];
cx q[17],q[3];
cx q[13],q[4];
rz(3.282522698386384) q[4];
cx q[13],q[4];
cx q[5],q[9];
rz(3.5144186456220368) q[9];
cx q[5],q[9];
cx q[5],q[11];
rz(3.0228479558715007) q[11];
cx q[5],q[11];
cx q[17],q[5];
rz(2.9796435980337352) q[5];
cx q[17],q[5];
cx q[13],q[6];
rz(2.8006768124818646) q[6];
cx q[13],q[6];
cx q[18],q[7];
rz(3.1063407041017843) q[7];
cx q[18],q[7];
cx q[8],q[10];
rz(3.1471831799492267) q[10];
cx q[8],q[10];
cx q[8],q[11];
rz(3.188159614482792) q[11];
cx q[8],q[11];
cx q[15],q[8];
rz(3.0014139913203692) q[8];
cx q[15],q[8];
cx q[9],q[10];
rz(3.020012602773967) q[10];
cx q[9],q[10];
cx q[19],q[9];
rz(3.4901965599510736) q[9];
cx q[19],q[9];
cx q[15],q[10];
rz(2.446021665144037) q[10];
cx q[15],q[10];
cx q[19],q[11];
rz(2.950849251535228) q[11];
cx q[19],q[11];
cx q[16],q[12];
rz(2.5149991915394745) q[12];
cx q[16],q[12];
cx q[14],q[13];
rz(3.2959830808348127) q[13];
cx q[14],q[13];
cx q[16],q[14];
rz(2.819998868366151) q[14];
cx q[16],q[14];
cx q[18],q[15];
rz(2.8602501149771977) q[15];
cx q[18],q[15];
cx q[17],q[16];
rz(2.76561808388856) q[16];
cx q[17],q[16];
cx q[19],q[18];
rz(2.9053250894181755) q[18];
cx q[19],q[18];
rx(0.6272288896806745) q[0];
rx(0.6272288896806745) q[1];
rx(0.6272288896806745) q[2];
rx(0.6272288896806745) q[3];
rx(0.6272288896806745) q[4];
rx(0.6272288896806745) q[5];
rx(0.6272288896806745) q[6];
rx(0.6272288896806745) q[7];
rx(0.6272288896806745) q[8];
rx(0.6272288896806745) q[9];
rx(0.6272288896806745) q[10];
rx(0.6272288896806745) q[11];
rx(0.6272288896806745) q[12];
rx(0.6272288896806745) q[13];
rx(0.6272288896806745) q[14];
rx(0.6272288896806745) q[15];
rx(0.6272288896806745) q[16];
rx(0.6272288896806745) q[17];
rx(0.6272288896806745) q[18];
rx(0.6272288896806745) q[19];
