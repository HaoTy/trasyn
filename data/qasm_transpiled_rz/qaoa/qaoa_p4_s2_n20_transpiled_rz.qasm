OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
h q[0];
h q[1];
h q[2];
cx q[0],q[2];
rz(1.643277180559323) q[2];
cx q[0],q[2];
h q[3];
h q[4];
h q[5];
cx q[3],q[5];
rz(1.520598991358991) q[5];
cx q[3],q[5];
h q[6];
cx q[0],q[6];
rz(1.2694711640149574) q[6];
cx q[0],q[6];
cx q[3],q[6];
rz(1.1836816582142677) q[6];
cx q[3],q[6];
h q[7];
cx q[7],q[0];
rz(-1.719613203603973) q[0];
h q[0];
rz(2.545561711957612) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[1],q[7];
rz(1.3436498320327632) q[7];
cx q[1],q[7];
cx q[2],q[7];
rz(-1.745273506407511) q[7];
h q[7];
rz(2.545561711957612) q[7];
h q[7];
rz(3*pi) q[7];
cx q[2],q[7];
h q[8];
cx q[4],q[8];
rz(1.2042055624896415) q[8];
cx q[4],q[8];
cx q[5],q[8];
rz(1.4258394385890925) q[8];
cx q[5],q[8];
h q[9];
cx q[1],q[9];
rz(1.4045337985992274) q[9];
cx q[1],q[9];
h q[10];
h q[11];
cx q[11],q[1];
rz(-1.677376386409347) q[1];
h q[1];
rz(2.545561711957612) q[1];
h q[1];
rz(3*pi) q[1];
cx q[11],q[1];
cx q[11],q[2];
rz(-1.8794161575214132) q[2];
h q[2];
rz(2.545561711957612) q[2];
h q[2];
rz(3*pi) q[2];
cx q[11],q[2];
cx q[0],q[2];
rz(12.1492132001122) q[2];
cx q[0],q[2];
cx q[9],q[11];
rz(-1.7828439408717962) q[11];
h q[11];
rz(2.545561711957612) q[11];
h q[11];
rz(3*pi) q[11];
cx q[9],q[11];
h q[12];
cx q[12],q[5];
rz(-1.8034482695133058) q[5];
h q[5];
rz(2.545561711957612) q[5];
h q[5];
rz(3*pi) q[5];
cx q[12],q[5];
cx q[12],q[6];
rz(-1.505935954991139) q[6];
h q[6];
rz(2.545561711957612) q[6];
h q[6];
rz(3*pi) q[6];
cx q[12],q[6];
cx q[0],q[6];
rz(10.814833008178343) q[6];
cx q[0],q[6];
cx q[7],q[0];
rz(-4.348719450692191) q[0];
h q[0];
rz(2.7068843393044455) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[1],q[7];
rz(4.796444255591531) q[7];
cx q[1],q[7];
cx q[2],q[7];
rz(-4.440319366864225) q[7];
h q[7];
rz(2.7068843393044455) q[7];
h q[7];
rz(3*pi) q[7];
cx q[2],q[7];
h q[13];
cx q[4],q[13];
rz(1.1944753536861636) q[13];
cx q[4],q[13];
cx q[10],q[13];
rz(1.7372567550288058) q[13];
cx q[10],q[13];
h q[14];
cx q[10],q[14];
rz(1.405675945498838) q[14];
cx q[10],q[14];
h q[15];
cx q[14],q[15];
rz(1.598653669024538) q[15];
cx q[14],q[15];
h q[16];
cx q[16],q[3];
rz(-1.7783570250623204) q[3];
h q[3];
rz(2.545561711957612) q[3];
h q[3];
rz(3*pi) q[3];
cx q[16],q[3];
cx q[16],q[10];
rz(-1.6766583427874107) q[10];
h q[10];
rz(2.545561711957612) q[10];
h q[10];
rz(3*pi) q[10];
cx q[16],q[10];
cx q[3],q[5];
rz(11.711287274883524) q[5];
cx q[3],q[5];
cx q[3],q[6];
rz(4.225403787980711) q[6];
cx q[3],q[6];
h q[17];
cx q[17],q[4];
rz(-1.5753179464394167) q[4];
h q[4];
rz(2.545561711957612) q[4];
h q[4];
rz(3*pi) q[4];
cx q[17],q[4];
cx q[17],q[12];
rz(-1.9377062812779102) q[12];
h q[12];
rz(2.545561711957612) q[12];
h q[12];
rz(3*pi) q[12];
cx q[17],q[12];
cx q[15],q[17];
rz(-1.937296172709564) q[17];
h q[17];
rz(2.545561711957612) q[17];
h q[17];
rz(3*pi) q[17];
cx q[15],q[17];
h q[18];
cx q[18],q[13];
rz(-1.5398871422751004) q[13];
h q[13];
rz(2.545561711957612) q[13];
h q[13];
rz(3*pi) q[13];
cx q[18],q[13];
cx q[18],q[14];
rz(-1.6289418556484283) q[14];
h q[14];
rz(2.545561711957612) q[14];
h q[14];
rz(3*pi) q[14];
cx q[18],q[14];
cx q[18],q[15];
rz(-1.7485182106542303) q[15];
h q[15];
rz(2.545561711957612) q[15];
h q[15];
rz(3*pi) q[15];
cx q[18],q[15];
h q[18];
rz(3.737623595221974) q[18];
h q[18];
h q[19];
cx q[19],q[8];
rz(-1.4141741841899633) q[8];
h q[8];
rz(2.545561711957612) q[8];
h q[8];
rz(3*pi) q[8];
cx q[19],q[8];
cx q[19],q[9];
rz(-1.7300160491290946) q[9];
h q[9];
rz(2.545561711957612) q[9];
h q[9];
rz(3*pi) q[9];
cx q[19],q[9];
cx q[1],q[9];
rz(11.296967848932734) q[9];
cx q[1],q[9];
cx q[11],q[1];
rz(-4.197946135816582) q[1];
h q[1];
rz(2.7068843393044455) q[1];
h q[1];
rz(3*pi) q[1];
cx q[11],q[1];
cx q[11],q[2];
rz(1.3640151765031066) q[2];
h q[2];
rz(2.7068843393044455) q[2];
h q[2];
rz(3*pi) q[2];
cx q[11],q[2];
cx q[0],q[2];
rz(10.890957917224021) q[2];
cx q[0],q[2];
cx q[19],q[16];
rz(-1.6473095539082339) q[16];
h q[16];
rz(2.545561711957612) q[16];
h q[16];
rz(3*pi) q[16];
cx q[19],q[16];
h q[19];
rz(3.737623595221974) q[19];
h q[19];
cx q[16],q[3];
rz(-4.558418033066973) q[3];
h q[3];
rz(2.7068843393044455) q[3];
h q[3];
rz(3*pi) q[3];
cx q[16],q[3];
cx q[4],q[8];
rz(10.581853542799156) q[8];
cx q[4],q[8];
cx q[4],q[13];
rz(10.547119490065683) q[13];
cx q[4],q[13];
cx q[10],q[13];
rz(6.201508000443317) q[13];
cx q[10],q[13];
cx q[10],q[14];
rz(11.30104498555351) q[14];
cx q[10],q[14];
cx q[14],q[15];
rz(11.989920071454165) q[15];
cx q[14],q[15];
cx q[16],q[10];
rz(-4.195382926174014) q[10];
h q[10];
rz(2.7068843393044455) q[10];
h q[10];
rz(3*pi) q[10];
cx q[16],q[10];
cx q[17],q[4];
rz(-3.8336267972074816) q[4];
h q[4];
rz(2.7068843393044455) q[4];
h q[4];
rz(3*pi) q[4];
cx q[17],q[4];
cx q[18],q[13];
rz(-3.7071489954368935) q[13];
h q[13];
rz(2.7068843393044455) q[13];
h q[13];
rz(3*pi) q[13];
cx q[18],q[13];
cx q[18],q[14];
rz(-4.025048762332734) q[14];
h q[14];
rz(2.7068843393044455) q[14];
h q[14];
rz(3*pi) q[14];
cx q[18],q[14];
cx q[5],q[8];
rz(5.089837561524547) q[8];
cx q[5],q[8];
cx q[12],q[5];
rz(-4.647986574247453) q[5];
h q[5];
rz(2.7068843393044455) q[5];
h q[5];
rz(3*pi) q[5];
cx q[12],q[5];
cx q[12],q[6];
rz(-3.5859530012601195) q[6];
h q[6];
rz(2.7068843393044455) q[6];
h q[6];
rz(3*pi) q[6];
cx q[12],q[6];
cx q[0],q[6];
rz(9.842800524862078) q[6];
cx q[0],q[6];
cx q[17],q[12];
rz(1.1559361647068611) q[12];
h q[12];
rz(2.7068843393044455) q[12];
h q[12];
rz(3*pi) q[12];
cx q[17],q[12];
cx q[15],q[17];
rz(1.1574001345894285) q[17];
h q[17];
rz(2.7068843393044455) q[17];
h q[17];
rz(3*pi) q[17];
cx q[15],q[17];
cx q[18],q[15];
rz(-4.451902029753206) q[15];
h q[15];
rz(2.7068843393044455) q[15];
h q[15];
rz(3*pi) q[15];
cx q[18],q[15];
h q[18];
rz(3.5763009678751416) q[18];
h q[18];
cx q[19],q[8];
rz(-3.2583898163071257) q[8];
h q[8];
rz(2.7068843393044455) q[8];
h q[8];
rz(3*pi) q[8];
cx q[19],q[8];
cx q[3],q[5];
rz(10.546966528105642) q[5];
cx q[3],q[5];
cx q[3],q[6];
cx q[4],q[8];
rz(9.659794848982088) q[8];
cx q[4],q[8];
cx q[4],q[13];
rz(9.632511204965674) q[13];
cx q[4],q[13];
cx q[10],q[13];
rz(4.871292674868022) q[13];
cx q[10],q[13];
cx q[10],q[14];
rz(10.224720671325073) q[14];
cx q[10],q[14];
cx q[14],q[15];
rz(10.765832955564179) q[15];
cx q[14],q[15];
cx q[17],q[4];
rz(-1.891328728877678) q[4];
h q[4];
rz(2.9729162963570275) q[4];
h q[4];
cx q[17],q[4];
cx q[18],q[13];
rz(-1.7919802495635446) q[13];
h q[13];
rz(2.9729162963570275) q[13];
h q[13];
cx q[18],q[13];
cx q[18],q[14];
rz(-2.041690933757012) q[14];
h q[14];
rz(2.9729162963570275) q[14];
h q[14];
cx q[18],q[14];
cx q[5],q[8];
rz(3.319060222010297) q[6];
cx q[3],q[6];
cx q[7],q[0];
rz(-2.2959346843354527) q[0];
h q[0];
rz(2.9729162963570275) q[0];
h q[0];
cx q[7],q[0];
cx q[1],q[7];
rz(3.7676132589050253) q[7];
cx q[1],q[7];
cx q[2],q[7];
rz(-2.367886538728845) q[7];
h q[7];
rz(2.9729162963570275) q[7];
h q[7];
cx q[2],q[7];
rz(3.9980740858433514) q[8];
cx q[5],q[8];
cx q[12],q[5];
rz(-2.531009402726424) q[5];
h q[5];
rz(2.9729162963570275) q[5];
h q[5];
cx q[12],q[5];
cx q[12],q[6];
rz(-1.6967806370476106) q[6];
h q[6];
rz(2.9729162963570275) q[6];
h q[6];
cx q[12],q[6];
cx q[17],q[12];
rz(-2.9074707791799117) q[12];
h q[12];
rz(2.9729162963570275) q[12];
h q[12];
cx q[17],q[12];
cx q[15],q[17];
rz(-2.9063208289154483) q[17];
h q[17];
rz(2.9729162963570275) q[17];
h q[17];
cx q[15],q[17];
cx q[18],q[15];
rz(-2.3769847356323277) q[15];
h q[15];
rz(2.9729162963570275) q[15];
h q[15];
cx q[18],q[15];
h q[18];
rz(2.9729162963570275) q[18];
h q[18];
cx q[9],q[11];
rz(-4.574435034704315) q[11];
h q[11];
rz(2.7068843393044446) q[11];
h q[11];
rz(3*pi) q[11];
cx q[9],q[11];
cx q[19],q[9];
rz(-4.3858546234665505) q[9];
h q[9];
rz(2.7068843393044455) q[9];
h q[9];
rz(3*pi) q[9];
cx q[19],q[9];
cx q[1],q[9];
rz(10.221518075157526) q[9];
cx q[1],q[9];
cx q[11],q[1];
rz(-2.1775020475853877) q[1];
h q[1];
rz(2.9729162963570275) q[1];
h q[1];
cx q[11],q[1];
cx q[11],q[2];
rz(-2.744024442275344) q[2];
h q[2];
rz(2.9729162963570275) q[2];
h q[2];
cx q[11],q[2];
cx q[0],q[2];
rz(12.069564979266534) q[2];
cx q[0],q[2];
cx q[0],q[6];
rz(10.753302841006999) q[6];
cx q[0],q[6];
cx q[19],q[16];
rz(-4.090616173354845) q[16];
h q[16];
rz(2.7068843393044446) q[16];
h q[16];
rz(3*pi) q[16];
cx q[19],q[16];
h q[19];
rz(3.5763009678751416) q[19];
h q[19];
cx q[16],q[3];
rz(-2.4606531960554188) q[3];
h q[3];
rz(2.9729162963570275) q[3];
h q[3];
cx q[16],q[3];
cx q[16],q[10];
rz(-2.175488643046508) q[10];
h q[10];
rz(2.9729162963570275) q[10];
h q[10];
cx q[16],q[10];
cx q[19],q[8];
rz(-1.4394793252853928) q[8];
h q[8];
rz(2.9729162963570275) q[8];
h q[8];
cx q[19],q[8];
cx q[3],q[5];
rz(11.63758515955309) q[5];
cx q[3],q[5];
cx q[3],q[6];
cx q[4],q[8];
rz(10.5234867427733) q[8];
cx q[4],q[8];
cx q[4],q[13];
rz(10.489224304828145) q[13];
cx q[4],q[13];
cx q[10],q[13];
rz(6.117304671067647) q[13];
cx q[10],q[13];
cx q[10],q[14];
rz(11.232913090661363) q[14];
cx q[10],q[14];
cx q[14],q[15];
rz(11.912434713540986) q[15];
cx q[14],q[15];
cx q[17],q[4];
rz(-3.909542774898861) q[4];
h q[4];
rz(2.4117223643835377) q[4];
h q[4];
rz(3*pi) q[4];
cx q[17],q[4];
cx q[18],q[13];
rz(-3.7847822735010914) q[13];
h q[13];
rz(2.4117223643835377) q[13];
h q[13];
rz(3*pi) q[13];
cx q[18],q[13];
cx q[18],q[14];
rz(-4.098365635616278) q[14];
h q[14];
rz(2.4117223643835377) q[14];
h q[14];
rz(3*pi) q[14];
cx q[18],q[14];
cx q[5],q[8];
rz(4.168031763808669) q[6];
cx q[3],q[6];
cx q[7],q[0];
rz(-4.417641562317887) q[0];
h q[0];
rz(2.4117223643835377) q[0];
h q[0];
rz(3*pi) q[0];
cx q[7],q[0];
cx q[1],q[7];
rz(4.731318712666045) q[7];
cx q[1],q[7];
cx q[2],q[7];
rz(-4.507997745850753) q[7];
h q[7];
rz(2.4117223643835377) q[7];
h q[7];
rz(3*pi) q[7];
cx q[2],q[7];
rz(5.020728359596392) q[8];
cx q[5],q[8];
cx q[12],q[5];
rz(1.5703400342979599) q[5];
h q[5];
rz(2.4117223643835377) q[5];
h q[5];
rz(3*pi) q[5];
cx q[12],q[5];
cx q[12],q[6];
rz(-3.6652318639492107) q[6];
h q[6];
rz(2.4117223643835377) q[6];
h q[6];
rz(3*pi) q[6];
cx q[12],q[6];
cx q[17],q[12];
rz(1.0975848355523068) q[12];
h q[12];
rz(2.4117223643835377) q[12];
h q[12];
rz(3*pi) q[12];
cx q[17],q[12];
cx q[15],q[17];
rz(1.0990289278279732) q[17];
h q[17];
rz(2.4117223643835377) q[17];
h q[17];
rz(3*pi) q[17];
cx q[15],q[17];
cx q[18],q[15];
rz(-4.519423140736819) q[15];
h q[15];
rz(2.4117223643835377) q[15];
h q[15];
rz(3*pi) q[15];
cx q[18],q[15];
h q[18];
rz(3.8714629427960485) q[18];
h q[18];
cx q[9],q[11];
rz(-2.473234571866552) q[11];
h q[11];
rz(2.9729162963570275) q[11];
h q[11];
cx q[9],q[11];
cx q[19],q[9];
rz(-2.3251044112958565) q[9];
h q[9];
rz(2.9729162963570275) q[9];
h q[9];
cx q[19],q[9];
cx q[1],q[9];
rz(11.228891312911003) q[9];
cx q[1],q[9];
cx q[11],q[1];
rz(-4.268915429368453) q[1];
h q[1];
rz(2.4117223643835377) q[1];
h q[1];
rz(3*pi) q[1];
cx q[11],q[1];
cx q[11],q[2];
rz(1.302838575565433) q[2];
h q[2];
rz(2.4117223643835377) q[2];
h q[2];
rz(3*pi) q[2];
cx q[11],q[2];
cx q[19],q[16];
rz(-2.0931942211831087) q[16];
h q[16];
rz(2.9729162963570275) q[16];
h q[16];
cx q[19],q[16];
h q[19];
rz(2.9729162963570275) q[19];
h q[19];
cx q[16],q[3];
rz(-4.624492882575125) q[3];
h q[3];
rz(2.4117223643835377) q[3];
h q[3];
rz(3*pi) q[3];
cx q[16],q[3];
cx q[16],q[10];
rz(-4.2663870226779705) q[10];
h q[10];
rz(2.4117223643835377) q[10];
h q[10];
rz(3*pi) q[10];
cx q[16],q[10];
cx q[19],q[8];
rz(-3.3421162925044103) q[8];
h q[8];
rz(2.4117223643835377) q[8];
h q[8];
rz(3*pi) q[8];
cx q[19],q[8];
cx q[9],q[11];
rz(-4.64029240729163) q[11];
h q[11];
rz(2.4117223643835377) q[11];
h q[11];
rz(3*pi) q[11];
cx q[9],q[11];
cx q[19],q[9];
rz(-4.454272518184952) q[9];
h q[9];
rz(2.4117223643835377) q[9];
h q[9];
rz(3*pi) q[9];
cx q[19],q[9];
cx q[19],q[16];
rz(-4.1630427802215415) q[16];
h q[16];
rz(2.4117223643835377) q[16];
h q[16];
rz(3*pi) q[16];
cx q[19],q[16];
h q[19];
rz(3.8714629427960485) q[19];
h q[19];
