OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[3];
rz(4.156514603158893) q[3];
cx q[0],q[3];
cx q[1],q[3];
rz(4.2460592498930945) q[3];
cx q[1],q[3];
h q[4];
cx q[2],q[4];
rz(4.342001100493026) q[4];
cx q[2],q[4];
h q[5];
cx q[0],q[5];
rz(4.458634729236196) q[5];
cx q[0],q[5];
h q[6];
h q[7];
h q[8];
cx q[4],q[8];
rz(4.320608040925396) q[8];
cx q[4],q[8];
h q[9];
cx q[9],q[0];
rz(-2.9719464652904426) q[0];
h q[0];
rz(2.448227589404887) q[0];
h q[0];
cx q[9],q[0];
cx q[7],q[9];
rz(4.578537726508901) q[9];
cx q[7],q[9];
h q[10];
cx q[10],q[3];
rz(-0.9194203869243536) q[3];
h q[3];
rz(2.448227589404887) q[3];
h q[3];
cx q[10],q[3];
cx q[0],q[3];
rz(6.702466794403683) q[3];
cx q[0],q[3];
h q[11];
cx q[2],q[11];
rz(4.407565401989229) q[11];
cx q[2],q[11];
cx q[5],q[11];
rz(4.296859462600884) q[11];
cx q[5],q[11];
cx q[10],q[11];
rz(-1.7676231382640002) q[11];
h q[11];
rz(2.448227589404887) q[11];
h q[11];
cx q[10],q[11];
h q[12];
cx q[1],q[12];
rz(4.363477049023618) q[12];
cx q[1],q[12];
cx q[12],q[9];
rz(-2.1999001407717955) q[9];
h q[9];
rz(2.448227589404887) q[9];
h q[9];
cx q[12],q[9];
h q[13];
cx q[13],q[4];
rz(-1.1665516841962589) q[4];
h q[4];
rz(2.448227589404887) q[4];
h q[4];
cx q[13],q[4];
cx q[7],q[13];
rz(4.539206928293591) q[13];
cx q[7],q[13];
h q[14];
cx q[14],q[12];
rz(-1.6815805877346577) q[12];
h q[12];
rz(2.448227589404887) q[12];
h q[12];
cx q[14],q[12];
h q[15];
cx q[15],q[2];
rz(-2.2017251419002246) q[2];
h q[2];
rz(2.448227589404887) q[2];
h q[2];
cx q[15],q[2];
cx q[2],q[4];
rz(6.72117743586351) q[4];
cx q[2],q[4];
cx q[2],q[11];
rz(6.72779112556864) q[11];
cx q[2],q[11];
h q[16];
cx q[6],q[16];
rz(4.673594026921785) q[16];
cx q[6],q[16];
cx q[16],q[13];
rz(-2.9533804869840568) q[13];
h q[13];
rz(2.448227589404887) q[13];
h q[13];
cx q[16],q[13];
cx q[14],q[16];
rz(-1.7872050704802165) q[16];
h q[16];
rz(2.448227589404887) q[16];
h q[16];
cx q[14],q[16];
h q[17];
cx q[17],q[1];
rz(-2.000696082061638) q[1];
h q[1];
rz(2.448227589404887) q[1];
h q[1];
cx q[17],q[1];
cx q[1],q[3];
rz(0.4283141543118374) q[3];
cx q[1],q[3];
cx q[1],q[12];
rz(6.723343786399752) q[12];
cx q[1],q[12];
cx q[6],q[17];
rz(3.9937250661245165) q[17];
cx q[6],q[17];
h q[18];
cx q[18],q[7];
rz(-2.867811754053199) q[7];
h q[7];
rz(2.448227589404887) q[7];
h q[7];
cx q[18],q[7];
cx q[8],q[18];
rz(5.031217425751598) q[18];
cx q[8],q[18];
cx q[18],q[10];
rz(-1.7689377963935553) q[10];
h q[10];
rz(2.448227589404887) q[10];
h q[10];
cx q[18],q[10];
h q[18];
rz(2.448227589404887) q[18];
h q[18];
cx q[10],q[3];
rz(-2.6005318015057197) q[3];
h q[3];
rz(0.2806470208088685) q[3];
h q[3];
rz(3*pi) q[3];
cx q[10],q[3];
h q[19];
cx q[19],q[8];
rz(-1.5958164457003932) q[8];
h q[8];
rz(2.448227589404887) q[8];
h q[8];
cx q[19],q[8];
cx q[15],q[19];
rz(4.327817363386951) q[19];
cx q[15],q[19];
cx q[4],q[8];
rz(6.719019446613339) q[8];
cx q[4],q[8];
cx q[13],q[4];
rz(-2.625460759479556) q[4];
h q[4];
rz(0.2806470208088685) q[4];
h q[4];
rz(3*pi) q[4];
cx q[13],q[4];
h q[20];
cx q[20],q[5];
rz(-2.36272740399823) q[5];
h q[5];
rz(2.448227589404887) q[5];
h q[5];
cx q[20],q[5];
cx q[0],q[5];
rz(6.732942658890181) q[5];
cx q[0],q[5];
cx q[20],q[15];
rz(-1.7614143411979972) q[15];
h q[15];
rz(2.448227589404887) q[15];
h q[15];
cx q[20],q[15];
cx q[15],q[2];
rz(-2.729882157771089) q[2];
h q[2];
rz(0.2806470208088685) q[2];
h q[2];
rz(3*pi) q[2];
cx q[15],q[2];
cx q[20],q[19];
rz(-1.4570192934867943) q[19];
h q[19];
rz(2.448227589404887) q[19];
h q[19];
cx q[20],q[19];
h q[20];
rz(2.448227589404887) q[20];
h q[20];
cx q[5],q[11];
rz(0.43343854115249353) q[11];
cx q[5],q[11];
cx q[10],q[11];
rz(-2.6860928412237035) q[11];
h q[11];
rz(0.2806470208088685) q[11];
h q[11];
rz(3*pi) q[11];
cx q[10],q[11];
cx q[20],q[5];
rz(-2.7461229927840565) q[5];
h q[5];
rz(0.2806470208088685) q[5];
h q[5];
rz(3*pi) q[5];
cx q[20],q[5];
cx q[9],q[0];
rz(-2.8075769508357715) q[0];
h q[0];
rz(0.2806470208088685) q[0];
h q[0];
rz(3*pi) q[0];
cx q[9],q[0];
cx q[7],q[9];
rz(0.46185236684200226) q[9];
cx q[7],q[9];
cx q[12],q[9];
rz(-2.729698063822103) q[9];
h q[9];
rz(0.2806470208088685) q[9];
h q[9];
rz(3*pi) q[9];
cx q[12],q[9];
cx q[7],q[13];
rz(0.4578849380840489) q[13];
cx q[7],q[13];
cx q[18],q[7];
rz(-2.7970725352934753) q[7];
h q[7];
rz(0.2806470208088685) q[7];
h q[7];
rz(3*pi) q[7];
cx q[18],q[7];
cx q[8],q[18];
rz(0.5075156774894348) q[18];
cx q[8],q[18];
cx q[18],q[10];
rz(-2.6862254551727394) q[10];
h q[10];
rz(0.2806470208088685) q[10];
h q[10];
rz(3*pi) q[10];
cx q[18],q[10];
h q[18];
rz(6.002538286370719) q[18];
h q[18];
cx q[19],q[8];
rz(-2.668762127284102) q[8];
h q[8];
rz(0.2806470208088685) q[8];
h q[8];
rz(3*pi) q[8];
cx q[19],q[8];
cx q[15],q[19];
rz(0.4365613678287317) q[19];
cx q[15],q[19];
cx q[20],q[15];
rz(-2.6854665391614594) q[15];
h q[15];
rz(0.2806470208088685) q[15];
h q[15];
rz(3*pi) q[15];
cx q[20],q[15];
cx q[20],q[19];
rz(-2.6547611957444026) q[19];
h q[19];
rz(0.2806470208088685) q[19];
h q[19];
rz(3*pi) q[19];
cx q[20],q[19];
h q[20];
rz(6.002538286370719) q[20];
h q[20];
h q[21];
cx q[21],q[6];
rz(-2.506553602552649) q[6];
h q[6];
rz(2.448227589404887) q[6];
h q[6];
cx q[21],q[6];
cx q[21],q[14];
rz(-1.7467848172765494) q[14];
h q[14];
rz(2.448227589404887) q[14];
h q[14];
cx q[21],q[14];
cx q[14],q[12];
rz(-2.6774134422572997) q[12];
h q[12];
rz(0.2806470208088685) q[12];
h q[12];
rz(3*pi) q[12];
cx q[14],q[12];
cx q[21],q[17];
rz(-1.3970758496422673) q[17];
h q[17];
rz(2.448227589404887) q[17];
h q[17];
cx q[21],q[17];
h q[21];
rz(2.448227589404887) q[21];
h q[21];
cx q[17],q[1];
rz(-2.7096036862073594) q[1];
h q[1];
rz(0.2806470208088685) q[1];
h q[1];
rz(3*pi) q[1];
cx q[17],q[1];
cx q[6],q[16];
rz(6.754626319993587) q[16];
cx q[6],q[16];
cx q[16],q[13];
rz(-2.8057041386988537) q[13];
h q[13];
rz(0.2806470208088685) q[13];
h q[13];
rz(3*pi) q[13];
cx q[16],q[13];
cx q[14],q[16];
rz(-2.688068136019034) q[16];
h q[16];
rz(0.2806470208088685) q[16];
h q[16];
rz(3*pi) q[16];
cx q[14],q[16];
cx q[6],q[17];
rz(0.4028603638289261) q[17];
cx q[6],q[17];
cx q[21],q[6];
rz(-2.760631220996669) q[6];
h q[6];
rz(0.2806470208088685) q[6];
h q[6];
rz(3*pi) q[6];
cx q[21],q[6];
cx q[21],q[14];
rz(-2.6839908103037997) q[14];
h q[14];
rz(0.2806470208088685) q[14];
h q[14];
rz(3*pi) q[14];
cx q[21],q[14];
cx q[21],q[17];
rz(-2.6487145006924075) q[17];
h q[17];
rz(0.2806470208088685) q[17];
h q[17];
rz(3*pi) q[17];
cx q[21],q[17];
h q[21];
rz(6.002538286370719) q[21];
h q[21];
