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
cx q[0],q[2];
rz(5.37258030534562) q[2];
cx q[0],q[2];
cx q[0],q[8];
rz(7.013236276020097) q[8];
cx q[0],q[8];
cx q[15],q[0];
rz(5.751502202335976) q[0];
cx q[15],q[0];
cx q[1],q[7];
rz(6.681628777539508) q[7];
cx q[1],q[7];
cx q[1],q[9];
rz(7.18182672158449) q[9];
cx q[1],q[9];
cx q[12],q[1];
rz(5.4087726233284545) q[1];
cx q[12],q[1];
cx q[2],q[4];
rz(6.226677068140181) q[4];
cx q[2],q[4];
cx q[13],q[2];
rz(5.659110323462778) q[2];
cx q[13],q[2];
cx q[3],q[5];
rz(5.630503842132896) q[5];
cx q[3],q[5];
cx q[3],q[7];
rz(6.278815372084234) q[7];
cx q[3],q[7];
cx q[15],q[3];
rz(5.8259163783473555) q[3];
cx q[15],q[3];
cx q[4],q[10];
rz(6.37230953778613) q[10];
cx q[4],q[10];
cx q[14],q[4];
rz(5.754985151893959) q[4];
cx q[14],q[4];
cx q[5],q[8];
rz(6.156422873869508) q[8];
cx q[5],q[8];
cx q[11],q[5];
rz(6.622682930834435) q[5];
cx q[11],q[5];
cx q[6],q[7];
rz(6.412134586382952) q[7];
cx q[6],q[7];
cx q[6],q[12];
rz(6.502993241606311) q[12];
cx q[6],q[12];
cx q[13],q[6];
rz(5.770888590426011) q[6];
cx q[13],q[6];
cx q[15],q[8];
rz(5.633415942553303) q[8];
cx q[15],q[8];
cx q[9],q[11];
rz(6.045240468117367) q[11];
cx q[9],q[11];
cx q[13],q[9];
rz(6.40014777491106) q[9];
cx q[13],q[9];
cx q[10],q[12];
rz(5.027201208329676) q[12];
cx q[10],q[12];
cx q[14],q[10];
rz(6.642146843126018) q[10];
cx q[14],q[10];
cx q[14],q[11];
rz(6.652238706052014) q[11];
cx q[14],q[11];
rx(3.942703626195262) q[0];
rx(3.942703626195262) q[1];
rx(3.942703626195262) q[2];
rx(3.942703626195262) q[3];
rx(3.942703626195262) q[4];
rx(3.942703626195262) q[5];
rx(3.942703626195262) q[6];
rx(3.942703626195262) q[7];
rx(3.942703626195262) q[8];
rx(3.942703626195262) q[9];
rx(3.942703626195262) q[10];
rx(3.942703626195262) q[11];
rx(3.942703626195262) q[12];
rx(3.942703626195262) q[13];
rx(3.942703626195262) q[14];
rx(3.942703626195262) q[15];
cx q[0],q[2];
rz(3.152628857715814) q[2];
cx q[0],q[2];
cx q[0],q[8];
rz(4.115365394866442) q[8];
cx q[0],q[8];
cx q[15],q[0];
rz(3.3749801376182496) q[0];
cx q[15],q[0];
cx q[1],q[7];
rz(3.920778192865075) q[7];
cx q[1],q[7];
cx q[1],q[9];
rz(4.214294228613727) q[9];
cx q[1],q[9];
cx q[12],q[1];
rz(3.1738665013833796) q[1];
cx q[12],q[1];
cx q[2],q[4];
rz(3.6538126369491657) q[4];
cx q[2],q[4];
cx q[13],q[2];
rz(3.3207646048574992) q[2];
cx q[13],q[2];
cx q[3],q[5];
rz(3.303978328351113) q[5];
cx q[3],q[5];
cx q[3],q[7];
rz(3.684407381422847) q[7];
cx q[3],q[7];
cx q[15],q[3];
rz(3.4186463585741644) q[3];
cx q[15],q[3];
cx q[4],q[10];
rz(3.7392697358350433) q[10];
cx q[4],q[10];
cx q[14],q[4];
rz(3.377023931598498) q[4];
cx q[14],q[4];
cx q[5],q[8];
rz(3.612587492298853) q[8];
cx q[5],q[8];
cx q[11],q[5];
rz(3.88618878390268) q[5];
cx q[11],q[5];
cx q[6],q[7];
rz(3.7626390649712413) q[7];
cx q[6],q[7];
cx q[6],q[12];
rz(3.8159549024554025) q[12];
cx q[6],q[12];
cx q[13],q[6];
rz(3.3863560655832) q[6];
cx q[13],q[6];
cx q[15],q[8];
rz(3.30568714819189) q[8];
cx q[15],q[8];
cx q[9],q[11];
rz(3.547345682791471) q[11];
cx q[9],q[11];
cx q[13],q[9];
rz(3.7556052068229167) q[9];
cx q[13],q[9];
cx q[10],q[12];
rz(2.9499604849376713) q[12];
cx q[10],q[12];
cx q[14],q[10];
rz(3.897610203050839) q[10];
cx q[14],q[10];
cx q[14],q[11];
rz(3.903532105838769) q[11];
cx q[14],q[11];
rx(0.3096758565685602) q[0];
rx(0.3096758565685602) q[1];
rx(0.3096758565685602) q[2];
rx(0.3096758565685602) q[3];
rx(0.3096758565685602) q[4];
rx(0.3096758565685602) q[5];
rx(0.3096758565685602) q[6];
rx(0.3096758565685602) q[7];
rx(0.3096758565685602) q[8];
rx(0.3096758565685602) q[9];
rx(0.3096758565685602) q[10];
rx(0.3096758565685602) q[11];
rx(0.3096758565685602) q[12];
rx(0.3096758565685602) q[13];
rx(0.3096758565685602) q[14];
rx(0.3096758565685602) q[15];
