OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
cx q[2],q[4];
rz(3.1668600451489257) q[4];
cx q[2],q[4];
h q[5];
cx q[1],q[5];
rz(3.4994850634922026) q[5];
cx q[1],q[5];
cx q[3],q[5];
rz(3.5933518443574686) q[5];
cx q[3],q[5];
h q[6];
cx q[2],q[6];
rz(2.8292413823059808) q[6];
cx q[2],q[6];
h q[7];
h q[8];
cx q[0],q[8];
rz(3.066225282595019) q[8];
cx q[0],q[8];
h q[9];
cx q[0],q[9];
rz(3.2058752108331703) q[9];
cx q[0],q[9];
h q[10];
cx q[8],q[10];
rz(3.318355744843117) q[10];
cx q[8],q[10];
h q[11];
cx q[7],q[11];
rz(3.0697702438397414) q[11];
cx q[7],q[11];
h q[12];
cx q[12],q[0];
rz(0.7217083260104955) q[0];
h q[0];
rz(2.4154989000301086) q[0];
h q[0];
rz(3*pi) q[0];
cx q[12],q[0];
cx q[3],q[12];
rz(3.632243067147936) q[12];
cx q[3],q[12];
cx q[6],q[12];
rz(-0.2395263801599814) q[12];
h q[12];
rz(2.4154989000301086) q[12];
h q[12];
rz(3*pi) q[12];
cx q[6],q[12];
h q[13];
cx q[1],q[13];
rz(2.7653807120290623) q[13];
cx q[1],q[13];
cx q[4],q[13];
rz(3.2833941764321715) q[13];
cx q[4],q[13];
h q[14];
cx q[7],q[14];
rz(3.208246917776418) q[14];
cx q[7],q[14];
cx q[14],q[8];
rz(-0.05816506184542369) q[8];
h q[8];
rz(2.4154989000301086) q[8];
h q[8];
rz(3*pi) q[8];
cx q[14],q[8];
cx q[0],q[8];
rz(7.84620980525841) q[8];
cx q[0],q[8];
h q[15];
cx q[9],q[15];
rz(3.261780494896603) q[15];
cx q[9],q[15];
cx q[11],q[15];
rz(3.1852344982172434) q[15];
cx q[11],q[15];
h q[16];
cx q[16],q[1];
rz(-0.04571776975793718) q[1];
h q[1];
rz(2.4154989000301086) q[1];
h q[1];
rz(3*pi) q[1];
cx q[16],q[1];
h q[17];
cx q[17],q[2];
rz(0.5991665204895424) q[2];
h q[2];
rz(2.4154989000301086) q[2];
h q[2];
rz(3*pi) q[2];
cx q[17],q[2];
cx q[17],q[7];
rz(-0.16022534462440063) q[7];
h q[7];
rz(2.4154989000301086) q[7];
h q[7];
rz(3*pi) q[7];
cx q[17],q[7];
cx q[17],q[14];
rz(0.022942003100403063) q[14];
h q[14];
rz(2.4154989000301086) q[14];
h q[14];
rz(3*pi) q[14];
cx q[17],q[14];
h q[17];
rz(3.8676864071494776) q[17];
h q[17];
h q[18];
cx q[18],q[11];
rz(0.11103444073407598) q[11];
h q[11];
rz(2.4154989000301086) q[11];
h q[11];
rz(3*pi) q[11];
cx q[18],q[11];
cx q[16],q[18];
rz(3.3148325965261263) q[18];
cx q[16],q[18];
cx q[7],q[11];
rz(7.84801686793632) q[11];
cx q[7],q[11];
cx q[7],q[14];
rz(7.918606082304912) q[14];
cx q[7],q[14];
h q[19];
cx q[19],q[5];
rz(-0.26307512027946167) q[5];
h q[5];
rz(2.4154989000301086) q[5];
h q[5];
rz(3*pi) q[5];
cx q[19],q[5];
cx q[1],q[5];
rz(8.067066262029073) q[5];
cx q[1],q[5];
cx q[10],q[19];
rz(3.2658796960693763) q[19];
cx q[10],q[19];
h q[20];
cx q[20],q[4];
rz(0.2602829612493718) q[4];
h q[4];
rz(2.4154989000301086) q[4];
h q[4];
rz(3*pi) q[4];
cx q[20],q[4];
cx q[2],q[4];
rz(7.897508905857102) q[4];
cx q[2],q[4];
cx q[20],q[9];
rz(0.137296512926957) q[9];
h q[9];
rz(2.4154989000301086) q[9];
h q[9];
rz(3*pi) q[9];
cx q[20],q[9];
cx q[0],q[9];
rz(7.917397092190715) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(-1.1722546180437887) q[0];
h q[0];
rz(2.0126461577472083) q[0];
h q[0];
rz(3*pi) q[0];
cx q[12],q[0];
cx q[20],q[19];
rz(0.26390923634008434) q[19];
h q[19];
rz(2.4154989000301086) q[19];
h q[19];
rz(3*pi) q[19];
cx q[20],q[19];
h q[20];
rz(3.8676864071494776) q[20];
h q[20];
h q[21];
cx q[21],q[3];
rz(0.14478566928578385) q[3];
h q[3];
rz(2.4154989000301086) q[3];
h q[3];
rz(3*pi) q[3];
cx q[21],q[3];
cx q[21],q[6];
rz(0.24390118392775495) q[6];
h q[6];
rz(2.4154989000301086) q[6];
h q[6];
rz(3*pi) q[6];
cx q[21],q[6];
cx q[2],q[6];
rz(7.725406013082857) q[6];
cx q[2],q[6];
cx q[17],q[2];
rz(-1.234720949254116) q[2];
h q[2];
rz(2.0126461577472083) q[2];
h q[2];
rz(3*pi) q[2];
cx q[17],q[2];
cx q[17],q[7];
rz(-1.621824954805307) q[7];
h q[7];
rz(2.0126461577472083) q[7];
h q[7];
rz(3*pi) q[7];
cx q[17],q[7];
cx q[21],q[16];
rz(0.556897129764975) q[16];
h q[16];
rz(2.4154989000301086) q[16];
h q[16];
rz(3*pi) q[16];
cx q[21],q[16];
h q[21];
rz(3.8676864071494776) q[21];
h q[21];
cx q[3],q[5];
rz(1.8317300411123318) q[5];
cx q[3],q[5];
cx q[19],q[5];
rz(-1.67425317000635) q[5];
h q[5];
rz(2.0126461577472083) q[5];
h q[5];
rz(3*pi) q[5];
cx q[19],q[5];
cx q[3],q[12];
rz(1.8515550469026094) q[12];
cx q[3],q[12];
cx q[21],q[3];
rz(-1.4663439829303364) q[3];
h q[3];
rz(2.0126461577472083) q[3];
h q[3];
rz(3*pi) q[3];
cx q[21],q[3];
cx q[6],q[12];
rz(-1.6622490756188402) q[12];
h q[12];
rz(2.0126461577472083) q[12];
h q[12];
rz(3*pi) q[12];
cx q[6],q[12];
cx q[21],q[6];
rz(-1.4158193269618162) q[6];
h q[6];
rz(2.0126461577472083) q[6];
h q[6];
rz(3*pi) q[6];
cx q[21],q[6];
h q[22];
cx q[22],q[10];
rz(-0.07804311014424492) q[10];
h q[10];
rz(2.4154989000301086) q[10];
h q[10];
rz(3*pi) q[10];
cx q[22],q[10];
cx q[22],q[13];
rz(-0.010498080610441107) q[13];
h q[13];
rz(2.4154989000301086) q[13];
h q[13];
rz(3*pi) q[13];
cx q[22],q[13];
cx q[1],q[13];
rz(7.69285269983704) q[13];
cx q[1],q[13];
cx q[16],q[1];
rz(-1.563454115014825) q[1];
h q[1];
rz(2.0126461577472083) q[1];
h q[1];
rz(3*pi) q[1];
cx q[16],q[1];
cx q[1],q[5];
rz(11.220179747473573) q[5];
cx q[1],q[5];
cx q[3],q[5];
cx q[4],q[13];
rz(1.6737274862822444) q[13];
cx q[4],q[13];
cx q[20],q[4];
rz(-1.4074686296333772) q[4];
h q[4];
rz(2.0126461577472083) q[4];
h q[4];
rz(3*pi) q[4];
cx q[20],q[4];
cx q[2],q[4];
rz(10.750919888817485) q[4];
cx q[2],q[4];
cx q[2],q[6];
rz(10.274615109356084) q[6];
cx q[2],q[6];
rz(5.069419573378467) q[5];
cx q[3],q[5];
cx q[8],q[10];
rz(7.974734637221243) q[10];
cx q[8],q[10];
cx q[10],q[19];
cx q[14],q[8];
rz(-1.5697991877588011) q[8];
h q[8];
rz(2.0126461577472083) q[8];
h q[8];
rz(3*pi) q[8];
cx q[14],q[8];
cx q[0],q[8];
rz(10.608946640120962) q[8];
cx q[0],q[8];
cx q[17],q[14];
rz(-1.5284544339348725) q[14];
h q[14];
rz(2.0126461577472092) q[14];
h q[14];
rz(3*pi) q[14];
cx q[17],q[14];
h q[17];
rz(4.270539149432378) q[17];
h q[17];
cx q[17],q[2];
rz(-4.147399457979392) q[2];
h q[2];
rz(1.609622746447192) q[2];
h q[2];
rz(3*pi) q[2];
cx q[17],q[2];
rz(1.6647993876087508) q[19];
cx q[10],q[19];
h q[23];
cx q[23],q[15];
rz(0.04445255783632973) q[15];
h q[15];
rz(2.4154989000301086) q[15];
h q[15];
rz(3*pi) q[15];
cx q[23],q[15];
cx q[23],q[18];
rz(0.016047459632124728) q[18];
h q[18];
rz(2.4154989000301086) q[18];
h q[18];
rz(3*pi) q[18];
cx q[23],q[18];
cx q[23],q[22];
rz(-0.003193846546125201) q[22];
h q[22];
rz(2.4154989000301086) q[22];
h q[22];
rz(3*pi) q[22];
cx q[23],q[22];
h q[23];
rz(3.8676864071494776) q[23];
h q[23];
cx q[22],q[10];
rz(-1.5799321276446658) q[10];
h q[10];
rz(2.0126461577472083) q[10];
h q[10];
rz(3*pi) q[10];
cx q[22],q[10];
cx q[22],q[13];
rz(-1.5455006928495538) q[13];
h q[13];
rz(2.0126461577472083) q[13];
h q[13];
rz(3*pi) q[13];
cx q[22],q[13];
cx q[1],q[13];
rz(10.184521918960819) q[13];
cx q[1],q[13];
cx q[4],q[13];
rz(4.632138300416952) q[13];
cx q[4],q[13];
cx q[8],q[10];
rz(10.964646598190058) q[10];
cx q[8],q[10];
cx q[9],q[15];
rz(7.945895105386029) q[15];
cx q[9],q[15];
cx q[11],q[15];
rz(1.6236900729700632) q[15];
cx q[11],q[15];
cx q[18],q[11];
rz(-1.4835488495675095) q[11];
h q[11];
rz(2.0126461577472083) q[11];
h q[11];
rz(3*pi) q[11];
cx q[18],q[11];
cx q[16],q[18];
rz(1.6897533866186207) q[18];
cx q[16],q[18];
cx q[20],q[9];
rz(-1.470161619855201) q[9];
h q[9];
rz(2.0126461577472083) q[9];
h q[9];
rz(3*pi) q[9];
cx q[20],q[9];
cx q[0],q[9];
rz(10.805961602834131) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(-3.9745202481573627) q[0];
h q[0];
rz(1.609622746447192) q[0];
h q[0];
rz(3*pi) q[0];
cx q[12],q[0];
cx q[20],q[19];
rz(-1.4056201167939157) q[19];
h q[19];
rz(2.0126461577472092) q[19];
h q[19];
rz(3*pi) q[19];
cx q[20],q[19];
h q[20];
rz(4.270539149432378) q[20];
h q[20];
cx q[19],q[5];
rz(0.9193548287978608) q[5];
h q[5];
rz(1.609622746447192) q[5];
h q[5];
rz(3*pi) q[5];
cx q[19],q[5];
cx q[10],q[19];
rz(4.6074292673429635) q[19];
cx q[10],q[19];
cx q[20],q[4];
rz(-4.62548872469173) q[4];
h q[4];
rz(1.609622746447192) q[4];
h q[4];
rz(3*pi) q[4];
cx q[20],q[4];
cx q[2],q[4];
rz(10.888515225935603) q[4];
cx q[2],q[4];
cx q[21],q[16];
rz(-1.256267993953144) q[16];
h q[16];
rz(2.0126461577472083) q[16];
h q[16];
rz(3*pi) q[16];
cx q[21],q[16];
h q[21];
rz(4.270539149432378) q[21];
h q[21];
cx q[16],q[1];
rz(1.2259976667046883) q[1];
h q[1];
rz(1.609622746447192) q[1];
h q[1];
rz(3*pi) q[1];
cx q[16],q[1];
cx q[1],q[5];
rz(11.372227144638668) q[5];
cx q[1],q[5];
cx q[23],q[15];
rz(-1.5174893152887097) q[15];
h q[15];
rz(2.0126461577472083) q[15];
h q[15];
rz(3*pi) q[15];
cx q[23],q[15];
cx q[23],q[18];
rz(-1.5319689638049248) q[18];
h q[18];
rz(2.0126461577472083) q[18];
h q[18];
rz(3*pi) q[18];
cx q[23],q[18];
cx q[23],q[22];
rz(-1.5417773210389325) q[22];
h q[22];
rz(2.0126461577472083) q[22];
h q[22];
rz(3*pi) q[22];
cx q[23],q[22];
h q[23];
rz(4.270539149432378) q[23];
h q[23];
cx q[22],q[10];
rz(1.1803938070303843) q[10];
h q[10];
rz(1.609622746447192) q[10];
h q[10];
rz(3*pi) q[10];
cx q[22],q[10];
cx q[22],q[13];
rz(1.2756848081998857) q[13];
h q[13];
rz(1.609622746447192) q[13];
h q[13];
rz(3*pi) q[13];
cx q[22],q[13];
cx q[1],q[13];
rz(10.304673578935326) q[13];
cx q[1],q[13];
cx q[3],q[12];
rz(5.124286431561644) q[12];
cx q[3],q[12];
cx q[21],q[3];
rz(1.4947556131887278) q[3];
h q[3];
rz(1.609622746447192) q[3];
h q[3];
rz(3*pi) q[3];
cx q[21],q[3];
cx q[3],q[5];
cx q[4],q[13];
rz(4.774796871416925) q[13];
cx q[4],q[13];
rz(5.225545341918747) q[5];
cx q[3],q[5];
cx q[6],q[12];
rz(0.9525768591641768) q[12];
h q[12];
rz(1.609622746447192) q[12];
h q[12];
rz(3*pi) q[12];
cx q[6],q[12];
cx q[21],q[6];
rz(-4.6485997659139615) q[6];
h q[6];
rz(1.609622746447192) q[6];
h q[6];
rz(3*pi) q[6];
cx q[21],q[6];
cx q[2],q[6];
rz(10.397541420035097) q[6];
cx q[2],q[6];
cx q[7],q[11];
rz(10.613947791329988) q[11];
cx q[7],q[11];
cx q[7],q[14];
rz(10.809307553389822) q[14];
cx q[7],q[14];
cx q[14],q[8];
rz(1.2084373083326163) q[8];
h q[8];
rz(1.609622746447192) q[8];
h q[8];
rz(3*pi) q[8];
cx q[14],q[8];
cx q[0],q[8];
rz(10.742169547172075) q[8];
cx q[0],q[8];
cx q[17],q[7];
rz(1.0644529679188803) q[7];
h q[7];
rz(1.609622746447192) q[7];
h q[7];
rz(3*pi) q[7];
cx q[17],q[7];
cx q[17],q[14];
rz(1.3228613225635897) q[14];
h q[14];
rz(1.609622746447192) q[14];
h q[14];
rz(3*pi) q[14];
cx q[17],q[14];
h q[17];
rz(4.673562560732394) q[17];
h q[17];
cx q[17],q[2];
rz(-0.8432764076252743) q[2];
h q[2];
rz(3.1279409765771806) q[2];
h q[2];
cx q[17],q[2];
cx q[8],q[10];
rz(11.10882419709878) q[10];
cx q[8],q[10];
cx q[9],q[15];
rz(10.88483151414868) q[15];
cx q[9],q[15];
cx q[11],q[15];
rz(4.49365684476972) q[15];
cx q[11],q[15];
cx q[18],q[11];
rz(1.4471401427153117) q[11];
h q[11];
rz(1.609622746447192) q[11];
h q[11];
rz(3*pi) q[11];
cx q[18],q[11];
cx q[16],q[18];
rz(4.676490912986926) q[18];
cx q[16],q[18];
cx q[20],q[9];
rz(1.4841900806503139) q[9];
h q[9];
rz(1.609622746447192) q[9];
h q[9];
rz(3*pi) q[9];
cx q[20],q[9];
cx q[0],q[9];
rz(10.945252090605738) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(-0.6650729394522799) q[0];
h q[0];
rz(3.1279409765771806) q[0];
h q[0];
cx q[12],q[0];
cx q[20],q[19];
rz(-4.620372857746448) q[19];
h q[19];
rz(1.609622746447192) q[19];
h q[19];
rz(3*pi) q[19];
cx q[20],q[19];
h q[20];
rz(4.673562560732394) q[20];
h q[20];
cx q[19],q[5];
rz(-2.0971705389789825) q[5];
h q[5];
rz(3.1279409765771806) q[5];
h q[5];
cx q[19],q[5];
cx q[10],q[19];
rz(4.749326860340874) q[19];
cx q[10],q[19];
cx q[20],q[4];
rz(-1.3360896586352018) q[4];
h q[4];
rz(3.1279409765771806) q[4];
h q[4];
cx q[20],q[4];
cx q[2],q[4];
rz(11.718300200476278) q[4];
cx q[2],q[4];
cx q[21],q[16];
rz(-4.207032159138446) q[16];
h q[16];
rz(1.609622746447192) q[16];
h q[16];
rz(3*pi) q[16];
cx q[21],q[16];
h q[21];
rz(4.673562560732394) q[21];
h q[21];
cx q[16],q[1];
rz(-1.7810838489619476) q[1];
h q[1];
rz(3.1279409765771806) q[1];
h q[1];
cx q[16],q[1];
cx q[1],q[5];
rz(12.289166978706639) q[5];
cx q[1],q[5];
cx q[23],q[15];
rz(1.3532079270531447) q[15];
h q[15];
rz(1.609622746447192) q[15];
h q[15];
rz(3*pi) q[15];
cx q[23],q[15];
cx q[23],q[18];
rz(1.3131346564530935) q[18];
h q[18];
rz(1.609622746447192) q[18];
h q[18];
rz(3*pi) q[18];
cx q[23],q[18];
cx q[23],q[22];
rz(1.2859894565443915) q[22];
h q[22];
rz(1.609622746447192) q[22];
h q[22];
rz(3*pi) q[22];
cx q[23],q[22];
h q[23];
rz(4.673562560732394) q[23];
h q[23];
cx q[22],q[10];
rz(-1.828092196376315) q[10];
h q[10];
rz(3.1279409765771806) q[10];
h q[10];
cx q[22],q[10];
cx q[22],q[13];
rz(-1.7298664645976753) q[13];
h q[13];
rz(3.1279409765771806) q[13];
h q[13];
cx q[22],q[13];
cx q[1],q[13];
rz(11.029262392779923) q[13];
cx q[1],q[13];
cx q[3],q[12];
rz(5.282101965621871) q[12];
cx q[3],q[12];
cx q[21],q[3];
rz(-1.504048812696964) q[3];
h q[3];
rz(3.1279409765771806) q[3];
h q[3];
cx q[21],q[3];
cx q[3],q[5];
cx q[4],q[13];
rz(5.63511627747064) q[13];
cx q[4],q[13];
rz(6.167080277525802) q[5];
cx q[3],q[5];
cx q[6],q[12];
rz(-2.062925351040418) q[12];
h q[12];
rz(3.1279409765771806) q[12];
h q[12];
cx q[6],q[12];
cx q[21],q[6];
rz(-1.359912463604814) q[6];
h q[6];
rz(3.1279409765771806) q[6];
h q[6];
cx q[21],q[6];
cx q[2],q[6];
rz(11.13886309352755) q[6];
cx q[2],q[6];
cx q[7],q[11];
rz(10.74732472165039) q[11];
cx q[7],q[11];
cx q[7],q[14];
rz(10.948701088284366) q[14];
cx q[7],q[14];
cx q[14],q[8];
rz(-1.7991850235767877) q[8];
h q[8];
rz(3.1279409765771806) q[8];
h q[8];
cx q[14],q[8];
cx q[0],q[8];
rz(11.545586064410188) q[8];
cx q[0],q[8];
cx q[17],q[7];
rz(-1.9476037307813296) q[7];
h q[7];
rz(3.1279409765771806) q[7];
h q[7];
cx q[17],q[7];
cx q[17],q[14];
rz(-1.6812370285620784) q[14];
h q[14];
rz(3.1279409765771806) q[14];
h q[14];
cx q[17],q[14];
h q[17];
rz(3.1279409765771806) q[17];
h q[17];
cx q[17],q[2];
rz(-3.004710256666366) q[2];
h q[2];
rz(2.985420346798332) q[2];
h q[2];
rz(3*pi) q[2];
cx q[17],q[2];
cx q[8],q[10];
rz(11.978304280839602) q[10];
cx q[8],q[10];
cx q[9],q[15];
rz(11.026551002980558) q[15];
cx q[9],q[15];
cx q[11],q[15];
rz(4.632050524419006) q[15];
cx q[11],q[15];
cx q[18],q[11];
rz(-1.553130723620705) q[11];
h q[11];
rz(3.1279409765771806) q[11];
h q[11];
cx q[18],q[11];
cx q[16],q[18];
rz(4.820515436365475) q[18];
cx q[16],q[18];
cx q[20],q[9];
rz(-1.5149397378890574) q[9];
h q[9];
rz(3.1279409765771806) q[9];
h q[9];
cx q[20],q[9];
cx q[0],q[9];
rz(11.785259871902142) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(-2.7943982180088214) q[0];
h q[0];
rz(2.985420346798332) q[0];
h q[0];
rz(3*pi) q[0];
cx q[12],q[0];
cx q[20],q[19];
rz(-1.3308162354555186) q[19];
h q[19];
rz(3.1279409765771806) q[19];
h q[19];
cx q[20],q[19];
h q[20];
rz(3.1279409765771806) q[20];
h q[20];
cx q[19],q[5];
rz(-4.484530114996674) q[5];
h q[5];
rz(2.985420346798332) q[5];
h q[5];
rz(3*pi) q[5];
cx q[19],q[5];
cx q[10],q[19];
rz(5.605057098438085) q[19];
cx q[10],q[19];
cx q[20],q[4];
rz(-3.5863182385982078) q[4];
h q[4];
rz(2.985420346798332) q[4];
h q[4];
rz(3*pi) q[4];
cx q[20],q[4];
cx q[21],q[16];
rz(-0.904745650653437) q[16];
h q[16];
rz(3.1279409765771806) q[16];
h q[16];
cx q[21],q[16];
h q[21];
rz(3.1279409765771806) q[21];
h q[21];
cx q[16],q[1];
rz(-4.111491156487379) q[1];
h q[1];
rz(2.985420346798332) q[1];
h q[1];
rz(3*pi) q[1];
cx q[16],q[1];
cx q[23],q[15];
rz(-1.6499558226097837) q[15];
h q[15];
rz(3.1279409765771806) q[15];
h q[15];
cx q[23],q[15];
cx q[23],q[18];
rz(-1.691263252284617) q[18];
h q[18];
rz(3.1279409765771806) q[18];
h q[18];
cx q[23],q[18];
cx q[23],q[22];
rz(-1.7192444581968815) q[22];
h q[22];
rz(3.1279409765771806) q[22];
h q[22];
cx q[23],q[22];
h q[23];
rz(3.1279409765771806) q[23];
h q[23];
cx q[22],q[10];
rz(-4.166969433537747) q[10];
h q[10];
rz(2.985420346798332) q[10];
h q[10];
rz(3*pi) q[10];
cx q[22],q[10];
cx q[22],q[13];
rz(-4.051045461289684) q[13];
h q[13];
rz(2.985420346798332) q[13];
h q[13];
rz(3*pi) q[13];
cx q[22],q[13];
cx q[3],q[12];
rz(6.233827232299122) q[12];
cx q[3],q[12];
cx q[21],q[3];
rz(-3.784540150124838) q[3];
h q[3];
rz(2.985420346798332) q[3];
h q[3];
rz(3*pi) q[3];
cx q[21],q[3];
cx q[6],q[12];
rz(-4.444114654156268) q[12];
h q[12];
rz(2.985420346798332) q[12];
h q[12];
rz(3*pi) q[12];
cx q[6],q[12];
cx q[21],q[6];
rz(-3.614433419133507) q[6];
h q[6];
rz(2.985420346798332) q[6];
h q[6];
rz(3*pi) q[6];
cx q[21],q[6];
cx q[7],q[11];
rz(11.551670094453407) q[11];
cx q[7],q[11];
cx q[7],q[14];
rz(11.7893303074585) q[14];
cx q[7],q[14];
cx q[14],q[8];
rz(-4.132853787497722) q[8];
h q[8];
rz(2.985420346798332) q[8];
h q[8];
rz(3*pi) q[8];
cx q[14],q[8];
cx q[17],q[7];
rz(-4.308014468752269) q[7];
h q[7];
rz(2.985420346798332) q[7];
h q[7];
rz(3*pi) q[7];
cx q[17],q[7];
cx q[17],q[14];
rz(-3.9936540090566397) q[14];
h q[14];
rz(2.985420346798332) q[14];
h q[14];
rz(3*pi) q[14];
cx q[17],q[14];
h q[17];
rz(3.2977649603812544) q[17];
h q[17];
cx q[9],q[15];
rz(11.881207162813403) q[15];
cx q[9],q[15];
cx q[11],q[15];
rz(5.466649998133665) q[15];
cx q[11],q[15];
cx q[18],q[11];
rz(-3.8424656038402456) q[11];
h q[11];
rz(2.985420346798332) q[11];
h q[11];
rz(3*pi) q[11];
cx q[18],q[11];
cx q[16],q[18];
rz(5.689072379994374) q[18];
cx q[16],q[18];
cx q[20],q[9];
rz(-3.7973933942479676) q[9];
h q[9];
rz(2.985420346798332) q[9];
h q[9];
rz(3*pi) q[9];
cx q[20],q[9];
cx q[20],q[19];
rz(-3.5800946538891467) q[19];
h q[19];
rz(2.985420346798332) q[19];
h q[19];
rz(3*pi) q[19];
cx q[20],q[19];
h q[20];
rz(3.2977649603812544) q[20];
h q[20];
cx q[21],q[16];
rz(-3.0772549829196136) q[16];
h q[16];
rz(2.985420346798332) q[16];
h q[16];
rz(3*pi) q[16];
cx q[21],q[16];
h q[21];
rz(3.2977649603812544) q[21];
h q[21];
cx q[23],q[15];
rz(-3.9567365783077206) q[15];
h q[15];
rz(2.985420346798332) q[15];
h q[15];
rz(3*pi) q[15];
cx q[23],q[15];
cx q[23],q[18];
rz(-4.005486750432226) q[18];
h q[18];
rz(2.985420346798332) q[18];
h q[18];
rz(3*pi) q[18];
cx q[23],q[18];
cx q[23],q[22];
rz(-4.0385095895488705) q[22];
h q[22];
rz(2.985420346798332) q[22];
h q[22];
rz(3*pi) q[22];
cx q[23],q[22];
h q[23];
rz(3.2977649603812544) q[23];
h q[23];
