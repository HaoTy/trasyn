OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
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
cx q[0],q[4];
rz(4.5232854883135865) q[4];
cx q[0],q[4];
cx q[0],q[7];
rz(4.102639186428637) q[7];
cx q[0],q[7];
cx q[10],q[0];
rz(4.897049700348952) q[0];
cx q[10],q[0];
cx q[1],q[3];
rz(3.9300420267694034) q[3];
cx q[1],q[3];
cx q[1],q[9];
rz(4.275823049294184) q[9];
cx q[1],q[9];
cx q[13],q[1];
rz(4.9166605353546275) q[1];
cx q[13],q[1];
cx q[2],q[3];
rz(5.30972712304886) q[3];
cx q[2],q[3];
cx q[2],q[6];
rz(4.698055918269044) q[6];
cx q[2],q[6];
cx q[11],q[2];
rz(4.8214448623992725) q[2];
cx q[11],q[2];
cx q[15],q[3];
rz(4.648103966101921) q[3];
cx q[15],q[3];
cx q[4],q[13];
rz(5.0565109494679055) q[13];
cx q[4],q[13];
cx q[14],q[4];
rz(4.698952866421946) q[4];
cx q[14],q[4];
cx q[5],q[11];
rz(5.266416914646666) q[11];
cx q[5],q[11];
cx q[5],q[12];
rz(5.518916571009732) q[12];
cx q[5],q[12];
cx q[14],q[5];
rz(4.213261070806788) q[5];
cx q[14],q[5];
cx q[6],q[10];
rz(4.73573268065451) q[10];
cx q[6],q[10];
cx q[12],q[6];
rz(5.334359777835887) q[6];
cx q[12],q[6];
cx q[7],q[10];
rz(4.609334677782583) q[10];
cx q[7],q[10];
cx q[14],q[7];
rz(4.37574541602889) q[7];
cx q[14],q[7];
cx q[8],q[9];
rz(4.889548165030778) q[9];
cx q[8],q[9];
cx q[8],q[13];
rz(4.160870780750885) q[13];
cx q[8],q[13];
cx q[15],q[8];
rz(4.669215501324726) q[8];
cx q[15],q[8];
cx q[12],q[9];
rz(5.195805610287969) q[9];
cx q[12],q[9];
cx q[15],q[11];
rz(4.562741728430646) q[11];
cx q[15],q[11];
rx(6.010101812970164) q[0];
rx(6.010101812970164) q[1];
rx(6.010101812970164) q[2];
rx(6.010101812970164) q[3];
rx(6.010101812970164) q[4];
rx(6.010101812970164) q[5];
rx(6.010101812970164) q[6];
rx(6.010101812970164) q[7];
rx(6.010101812970164) q[8];
rx(6.010101812970164) q[9];
rx(6.010101812970164) q[10];
rx(6.010101812970164) q[11];
rx(6.010101812970164) q[12];
rx(6.010101812970164) q[13];
rx(6.010101812970164) q[14];
rx(6.010101812970164) q[15];
cx q[0],q[4];
rz(0.36727018070207534) q[4];
cx q[0],q[4];
cx q[0],q[7];
rz(0.33311561678960744) q[7];
cx q[0],q[7];
cx q[10],q[0];
rz(0.3976181324395583) q[0];
cx q[10],q[0];
cx q[1],q[3];
rz(0.31910151350550425) q[3];
cx q[1],q[3];
cx q[1],q[9];
rz(0.34717735770196945) q[9];
cx q[1],q[9];
cx q[13],q[1];
rz(0.39921044292600916) q[1];
cx q[13],q[1];
cx q[2],q[3];
rz(0.43112565965583605) q[3];
cx q[2],q[3];
cx q[2],q[6];
rz(0.3814607436362428) q[6];
cx q[2],q[6];
cx q[11],q[2];
rz(0.3914793639343489) q[2];
cx q[11],q[2];
cx q[15],q[3];
rz(0.37740487262252076) q[3];
cx q[15],q[3];
cx q[4],q[13];
rz(0.410565659614265) q[13];
cx q[4],q[13];
cx q[14],q[4];
rz(0.3815335717411784) q[4];
cx q[14],q[4];
cx q[5],q[11];
rz(0.4276090679865252) q[11];
cx q[5],q[11];
cx q[5],q[12];
rz(0.4481108900933252) q[12];
cx q[5],q[12];
cx q[14],q[5];
rz(0.3420976099824171) q[5];
cx q[14],q[5];
cx q[6],q[10];
rz(0.3845199251460834) q[10];
cx q[6],q[10];
cx q[12],q[6];
rz(0.43312571903704844) q[6];
cx q[12],q[6];
cx q[7],q[10];
rz(0.37425698298267346) q[10];
cx q[7],q[10];
cx q[14],q[7];
rz(0.3552905987922454) q[7];
cx q[14],q[7];
cx q[8],q[9];
rz(0.39700904193687675) q[9];
cx q[8],q[9];
cx q[8],q[13];
rz(0.33784375703734487) q[13];
cx q[8],q[13];
cx q[15],q[8];
rz(0.3791190331317812) q[8];
cx q[15],q[8];
cx q[12],q[9];
rz(0.42187575166624647) q[9];
cx q[12],q[9];
cx q[15],q[11];
rz(0.3704738476992296) q[11];
cx q[15],q[11];
rx(6.202002411171408) q[0];
rx(6.202002411171408) q[1];
rx(6.202002411171408) q[2];
rx(6.202002411171408) q[3];
rx(6.202002411171408) q[4];
rx(6.202002411171408) q[5];
rx(6.202002411171408) q[6];
rx(6.202002411171408) q[7];
rx(6.202002411171408) q[8];
rx(6.202002411171408) q[9];
rx(6.202002411171408) q[10];
rx(6.202002411171408) q[11];
rx(6.202002411171408) q[12];
rx(6.202002411171408) q[13];
rx(6.202002411171408) q[14];
rx(6.202002411171408) q[15];
