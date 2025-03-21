OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
h q[0];
h q[1];
cx q[0],q[1];
rz(2.7537411845040896) q[1];
cx q[0],q[1];
h q[2];
cx q[1],q[2];
rz(3.160686130168731) q[2];
cx q[1],q[2];
h q[3];
h q[4];
h q[5];
cx q[2],q[5];
rz(2.6401597278126507) q[5];
cx q[2],q[5];
h q[6];
cx q[4],q[6];
rz(2.4169738907413674) q[6];
cx q[4],q[6];
h q[7];
cx q[4],q[7];
rz(3.255127045304127) q[7];
cx q[4],q[7];
h q[8];
cx q[6],q[8];
rz(2.645572954910707) q[8];
cx q[6],q[8];
h q[9];
h q[10];
cx q[5],q[10];
rz(2.483613283497572) q[10];
cx q[5],q[10];
cx q[9],q[10];
rz(2.8393122757621434) q[10];
cx q[9],q[10];
h q[11];
cx q[3],q[11];
rz(2.6161948295435877) q[11];
cx q[3],q[11];
cx q[11],q[4];
rz(-0.7089087639540175) q[4];
h q[4];
rz(2.9622623818865153) q[4];
h q[4];
rz(3*pi) q[4];
cx q[11],q[4];
h q[12];
cx q[12],q[6];
rz(-0.5736731662295735) q[6];
h q[6];
rz(2.9622623818865153) q[6];
h q[6];
rz(3*pi) q[6];
cx q[12],q[6];
cx q[4],q[6];
rz(11.266691007601764) q[6];
cx q[4],q[6];
cx q[9],q[12];
rz(2.3125911583386882) q[12];
cx q[9],q[12];
h q[13];
h q[14];
cx q[8],q[14];
rz(2.5487363377312615) q[14];
cx q[8],q[14];
cx q[14],q[9];
rz(-0.46599443473448154) q[9];
h q[9];
rz(2.9622623818865153) q[9];
h q[9];
rz(3*pi) q[9];
cx q[14],q[9];
cx q[14],q[10];
rz(-0.6988083622784593) q[10];
h q[10];
rz(2.9622623818865153) q[10];
h q[10];
rz(3*pi) q[10];
cx q[14],q[10];
h q[15];
cx q[3],q[15];
rz(2.8461762133977047) q[15];
cx q[3],q[15];
cx q[13],q[15];
rz(2.6723492169529846) q[15];
cx q[13],q[15];
h q[16];
cx q[0],q[16];
rz(2.2856101622751717) q[16];
cx q[0],q[16];
cx q[13],q[16];
rz(2.6685001876310617) q[16];
cx q[13],q[16];
h q[17];
cx q[17],q[3];
rz(-0.3437640157614119) q[3];
h q[3];
rz(2.9622623818865144) q[3];
h q[3];
rz(3*pi) q[3];
cx q[17],q[3];
cx q[7],q[17];
rz(2.2565841983771127) q[17];
cx q[7],q[17];
h q[18];
cx q[18],q[1];
rz(-0.6028330022919901) q[1];
h q[1];
rz(2.9622623818865153) q[1];
h q[1];
rz(3*pi) q[1];
cx q[18],q[1];
h q[19];
cx q[19],q[8];
rz(-0.3188392274045242) q[8];
h q[8];
rz(2.9622623818865153) q[8];
h q[8];
rz(3*pi) q[8];
cx q[19],q[8];
cx q[19],q[11];
rz(-1.0037877791322667) q[11];
h q[11];
rz(2.9622623818865153) q[11];
h q[11];
rz(3*pi) q[11];
cx q[19],q[11];
cx q[19],q[12];
rz(-0.619516892373106) q[12];
h q[12];
rz(2.9622623818865153) q[12];
h q[12];
rz(3*pi) q[12];
cx q[19],q[12];
h q[19];
rz(3.320922925293072) q[19];
h q[19];
cx q[3],q[11];
rz(11.677460312130655) q[11];
cx q[3],q[11];
cx q[6],q[8];
rz(11.738034427471987) q[8];
cx q[6],q[8];
cx q[12],q[6];
rz(-4.13004082996839) q[6];
h q[6];
rz(1.4826818206059311) q[6];
h q[6];
rz(3*pi) q[6];
cx q[12],q[6];
cx q[8],q[14];
rz(-pi) q[14];
h q[14];
rz(2.9622623818865144) q[14];
h q[14];
rz(8.3967764669577) q[14];
cx q[8],q[14];
cx q[19],q[8];
rz(-3.6046042947784582) q[8];
h q[8];
rz(1.4826818206059311) q[8];
h q[8];
rz(3*pi) q[8];
cx q[19],q[8];
h q[20];
cx q[20],q[2];
rz(-0.35974649371298106) q[2];
h q[2];
rz(2.9622623818865153) q[2];
h q[2];
rz(3*pi) q[2];
cx q[20],q[2];
cx q[18],q[20];
rz(2.4556789969223605) q[20];
cx q[18],q[20];
h q[21];
cx q[21],q[13];
rz(-0.2503076622538547) q[13];
h q[13];
rz(2.9622623818865153) q[13];
h q[13];
rz(3*pi) q[13];
cx q[21],q[13];
cx q[21],q[15];
rz(-0.9195422834429126) q[15];
h q[15];
rz(2.9622623818865153) q[15];
h q[15];
rz(3*pi) q[15];
cx q[21],q[15];
cx q[21],q[16];
rz(-0.1643542057522165) q[16];
h q[16];
rz(2.9622623818865153) q[16];
h q[16];
rz(3*pi) q[16];
cx q[21],q[16];
h q[21];
rz(3.320922925293072) q[21];
h q[21];
cx q[3],q[15];
rz(12.151653906781537) q[15];
cx q[3],q[15];
cx q[13],q[15];
rz(5.510058510445458) q[15];
cx q[13],q[15];
h q[22];
cx q[22],q[0];
rz(-0.8128033582993566) q[0];
h q[0];
rz(2.9622623818865153) q[0];
h q[0];
rz(3*pi) q[0];
cx q[22],q[0];
cx q[0],q[1];
rz(11.961064138652631) q[1];
cx q[0],q[1];
cx q[0],q[16];
rz(10.995835003675634) q[16];
cx q[0],q[16];
cx q[1],q[2];
rz(0.23376442375429995) q[2];
cx q[1],q[2];
cx q[13],q[16];
rz(5.50212228091465) q[16];
cx q[13],q[16];
cx q[21],q[13];
rz(-3.4633005567052035) q[13];
h q[13];
rz(1.4826818206059311) q[13];
h q[13];
rz(3*pi) q[13];
cx q[21],q[13];
cx q[21],q[15];
rz(1.4400044950891866) q[15];
h q[15];
rz(1.4826818206059311) q[15];
h q[15];
rz(3*pi) q[15];
cx q[21],q[15];
cx q[21],q[16];
rz(-3.2860750011922666) q[16];
h q[16];
rz(1.4826818206059311) q[16];
h q[16];
rz(3*pi) q[16];
cx q[21],q[16];
h q[21];
rz(4.800503486573655) q[21];
h q[21];
cx q[22],q[5];
rz(-0.571181406998913) q[5];
h q[5];
rz(2.9622623818865153) q[5];
h q[5];
rz(3*pi) q[5];
cx q[22],q[5];
cx q[2],q[5];
rz(11.726873012694533) q[5];
cx q[2],q[5];
cx q[22],q[18];
rz(-0.1714173306200557) q[18];
h q[18];
rz(2.9622623818865153) q[18];
h q[18];
rz(3*pi) q[18];
cx q[22],q[18];
h q[22];
rz(3.320922925293072) q[22];
h q[22];
cx q[18],q[1];
rz(-4.190164859357825) q[1];
h q[1];
rz(1.4826818206059311) q[1];
h q[1];
rz(3*pi) q[1];
cx q[18],q[1];
cx q[22],q[0];
rz(-4.623098152551745) q[0];
h q[0];
rz(1.4826818206059311) q[0];
h q[0];
rz(3*pi) q[0];
cx q[22],q[0];
cx q[5],q[10];
rz(11.404093316722843) q[10];
cx q[5],q[10];
cx q[22],q[5];
rz(-4.124903126008226) q[5];
h q[5];
rz(1.4826818206059311) q[5];
h q[5];
rz(3*pi) q[5];
cx q[22],q[5];
cx q[9],q[10];
rz(5.854315996437641) q[10];
cx q[9],q[10];
cx q[9],q[12];
rz(4.768281223257953) q[12];
cx q[9],q[12];
cx q[14],q[9];
rz(-3.908020404230395) q[9];
h q[9];
rz(1.4826818206059311) q[9];
h q[9];
rz(3*pi) q[9];
cx q[14],q[9];
cx q[14],q[10];
rz(-4.388054358878207) q[10];
h q[10];
rz(1.4826818206059311) q[10];
h q[10];
rz(3*pi) q[10];
cx q[14],q[10];
h q[14];
rz(4.800503486573655) q[14];
h q[14];
h q[23];
cx q[23],q[7];
rz(-0.7236038564041611) q[7];
h q[7];
rz(2.9622623818865153) q[7];
h q[7];
rz(3*pi) q[7];
cx q[23],q[7];
cx q[23],q[17];
rz(-0.6633960667552321) q[17];
h q[17];
rz(2.9622623818865153) q[17];
h q[17];
rz(3*pi) q[17];
cx q[23],q[17];
cx q[17],q[3];
rz(-3.6559961717590865) q[3];
h q[3];
rz(1.4826818206059311) q[3];
h q[3];
rz(3*pi) q[3];
cx q[17],q[3];
cx q[23],q[20];
rz(-0.5050538643188975) q[20];
h q[20];
rz(2.9622623818865153) q[20];
h q[20];
rz(3*pi) q[20];
cx q[23],q[20];
h q[23];
rz(3.320922925293072) q[23];
h q[23];
cx q[20],q[2];
rz(-3.6889500941343436) q[2];
h q[2];
rz(1.4826818206059311) q[2];
h q[2];
rz(3*pi) q[2];
cx q[20],q[2];
cx q[18],q[20];
rz(5.063310913886548) q[20];
cx q[18],q[20];
cx q[22],q[18];
rz(-3.300638304162361) q[18];
h q[18];
rz(1.4826818206059311) q[18];
h q[18];
rz(3*pi) q[18];
cx q[22],q[18];
h q[22];
rz(4.800503486573655) q[22];
h q[22];
cx q[4],q[7];
rz(0.4284900849346833) q[7];
cx q[4],q[7];
cx q[11],q[4];
rz(-4.408880156591421) q[4];
h q[4];
rz(1.4826818206059311) q[4];
h q[4];
rz(3*pi) q[4];
cx q[11],q[4];
cx q[19],q[11];
rz(1.2663005466886483) q[11];
h q[11];
rz(1.4826818206059311) q[11];
h q[11];
rz(3*pi) q[11];
cx q[19],q[11];
cx q[19],q[12];
rz(-4.224565008090484) q[12];
h q[12];
rz(1.4826818206059311) q[12];
h q[12];
rz(3*pi) q[12];
cx q[19],q[12];
h q[19];
rz(4.800503486573655) q[19];
h q[19];
cx q[7],q[17];
rz(4.652801695199742) q[17];
cx q[7],q[17];
cx q[23],q[7];
rz(-4.439179646906171) q[7];
h q[7];
rz(1.4826818206059311) q[7];
h q[7];
rz(3*pi) q[7];
cx q[23],q[7];
cx q[23],q[17];
rz(-4.315038519823815) q[17];
h q[17];
rz(1.4826818206059311) q[17];
h q[17];
rz(3*pi) q[17];
cx q[23],q[17];
cx q[23],q[20];
rz(-3.9885561893850574) q[20];
h q[20];
rz(1.4826818206059311) q[20];
h q[20];
rz(3*pi) q[20];
cx q[23],q[20];
h q[23];
rz(4.800503486573655) q[23];
h q[23];
