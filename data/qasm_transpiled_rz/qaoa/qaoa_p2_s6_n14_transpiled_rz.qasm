OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
h q[0];
h q[1];
h q[2];
cx q[0],q[2];
rz(5.669559814321269) q[2];
cx q[0],q[2];
h q[3];
cx q[2],q[3];
rz(5.756537105904038) q[3];
cx q[2],q[3];
h q[4];
h q[5];
h q[6];
cx q[4],q[6];
rz(5.280131115715001) q[6];
cx q[4],q[6];
cx q[5],q[6];
rz(4.710283963111986) q[6];
cx q[5],q[6];
h q[7];
cx q[1],q[7];
rz(5.348690177755056) q[7];
cx q[1],q[7];
h q[8];
cx q[1],q[8];
rz(5.44567807932184) q[8];
cx q[1],q[8];
cx q[5],q[8];
rz(4.989021497092828) q[8];
cx q[5],q[8];
h q[9];
cx q[9],q[2];
rz(-3.783851622422509) q[2];
h q[2];
rz(1.9497263372198201) q[2];
h q[2];
rz(3*pi) q[2];
cx q[9],q[2];
cx q[3],q[9];
rz(5.520681131624359) q[9];
cx q[3],q[9];
cx q[9],q[6];
rz(-4.0133538463574325) q[6];
h q[6];
rz(1.9497263372198201) q[6];
h q[6];
rz(3*pi) q[6];
cx q[9],q[6];
h q[9];
rz(4.333458969959766) q[9];
h q[9];
h q[10];
cx q[10],q[5];
rz(-4.609331790414164) q[5];
h q[5];
rz(1.9497263372198201) q[5];
h q[5];
rz(3*pi) q[5];
cx q[10],q[5];
cx q[7],q[10];
rz(6.261553408205374) q[10];
cx q[7],q[10];
h q[11];
cx q[0],q[11];
rz(5.658715229444201) q[11];
cx q[0],q[11];
cx q[11],q[1];
rz(-4.39834168055108) q[1];
h q[1];
rz(1.9497263372198201) q[1];
h q[1];
rz(3*pi) q[1];
cx q[11],q[1];
cx q[11],q[10];
rz(-3.836182686901887) q[10];
h q[10];
rz(1.9497263372198201) q[10];
h q[10];
rz(3*pi) q[10];
cx q[11],q[10];
h q[12];
cx q[12],q[0];
rz(-4.054440139781924) q[0];
h q[0];
rz(1.9497263372198201) q[0];
h q[0];
rz(3*pi) q[0];
cx q[12],q[0];
cx q[0],q[2];
rz(8.92828168894307) q[2];
cx q[0],q[2];
cx q[0],q[11];
rz(-pi) q[11];
h q[11];
rz(1.9497263372198201) q[11];
h q[11];
rz(5.781629564572565) q[11];
cx q[0],q[11];
cx q[4],q[12];
rz(4.76305702921617) q[12];
cx q[4],q[12];
cx q[12],q[8];
rz(-3.3753736549087914) q[8];
h q[8];
rz(1.9497263372198201) q[8];
h q[8];
rz(3*pi) q[8];
cx q[12],q[8];
h q[12];
rz(4.333458969959766) q[12];
h q[12];
cx q[12],q[0];
rz(-0.6360963532844988) q[0];
h q[0];
rz(1.8913092489354497) q[0];
h q[0];
rz(3*pi) q[0];
cx q[12],q[0];
h q[13];
cx q[13],q[3];
rz(-4.505327688209951) q[3];
h q[3];
rz(1.9497263372198201) q[3];
h q[3];
rz(3*pi) q[3];
cx q[13],q[3];
cx q[13],q[4];
rz(-3.7187092822275143) q[4];
h q[4];
rz(1.9497263372198201) q[4];
h q[4];
rz(3*pi) q[4];
cx q[13],q[4];
cx q[13],q[7];
rz(-4.003759338280143) q[7];
h q[7];
rz(1.9497263372198201) q[7];
h q[7];
rz(3*pi) q[7];
cx q[13],q[7];
h q[13];
rz(4.333458969959766) q[13];
h q[13];
cx q[1],q[7];
rz(8.778582039853) q[7];
cx q[1],q[7];
cx q[1],q[8];
rz(8.823831116661594) q[8];
cx q[1],q[8];
cx q[11],q[1];
rz(-0.7965413879895928) q[1];
h q[1];
rz(1.8913092489354497) q[1];
h q[1];
rz(3*pi) q[1];
cx q[11],q[1];
cx q[2],q[3];
rz(8.96886038055155) q[3];
cx q[2],q[3];
cx q[4],q[6];
rz(8.746596253852267) q[6];
cx q[4],q[6];
cx q[4],q[12];
rz(2.222173383247228) q[12];
cx q[4],q[12];
cx q[5],q[6];
rz(2.1975524513268803) q[6];
cx q[5],q[6];
cx q[5],q[8];
rz(2.327595641052478) q[8];
cx q[5],q[8];
cx q[10],q[5];
rz(-0.8949774561261341) q[5];
h q[5];
rz(1.8913092489354497) q[5];
h q[5];
rz(3*pi) q[5];
cx q[10],q[5];
cx q[12],q[8];
rz(-0.31928228641856826) q[8];
h q[8];
rz(1.8913092489354497) q[8];
h q[8];
rz(3*pi) q[8];
cx q[12],q[8];
h q[12];
rz(4.3918760582441365) q[12];
h q[12];
cx q[7],q[10];
rz(2.921287155737612) q[10];
cx q[7],q[10];
cx q[11],q[10];
rz(-0.5342697533697027) q[10];
h q[10];
rz(1.8913092489354497) q[10];
h q[10];
rz(3*pi) q[10];
cx q[11],q[10];
h q[11];
rz(4.3918760582441365) q[11];
h q[11];
cx q[9],q[2];
rz(-0.5098550344423902) q[2];
h q[2];
rz(1.8913092489354497) q[2];
h q[2];
rz(3*pi) q[2];
cx q[9],q[2];
cx q[3],q[9];
rz(2.575637997370294) q[9];
cx q[3],q[9];
cx q[13],q[3];
rz(-0.8464550163885685) q[3];
h q[3];
rz(1.8913092489354497) q[3];
h q[3];
rz(3*pi) q[3];
cx q[13],q[3];
cx q[13],q[4];
rz(-0.4794632978715656) q[4];
h q[4];
rz(1.8913092489354497) q[4];
h q[4];
rz(3*pi) q[4];
cx q[13],q[4];
cx q[13],q[7];
rz(-0.6124515538525195) q[7];
h q[7];
rz(1.8913092489354497) q[7];
h q[7];
rz(3*pi) q[7];
cx q[13],q[7];
h q[13];
rz(4.3918760582441365) q[13];
h q[13];
cx q[9],q[6];
rz(-0.6169278094051878) q[6];
h q[6];
rz(1.8913092489354497) q[6];
h q[6];
rz(3*pi) q[6];
cx q[9],q[6];
h q[9];
rz(4.3918760582441365) q[9];
h q[9];
