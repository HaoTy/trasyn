OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.5111843265150166) q[3];
cx q[2],q[3];
h q[4];
cx q[3],q[4];
rz(0.4954136897216937) q[4];
cx q[3],q[4];
h q[5];
h q[6];
h q[7];
cx q[2],q[7];
rz(0.3661414751819321) q[7];
cx q[2],q[7];
h q[8];
cx q[4],q[8];
rz(0.4191366185745173) q[8];
cx q[4],q[8];
cx q[6],q[8];
rz(0.44170864068315996) q[8];
cx q[6],q[8];
h q[9];
cx q[9],q[4];
rz(0.6042230473742087) q[4];
h q[4];
rz(1.1226334665814068) q[4];
h q[4];
cx q[9],q[4];
h q[10];
cx q[1],q[10];
rz(0.5227677274482156) q[10];
cx q[1],q[10];
cx q[5],q[10];
rz(0.5388900714504642) q[10];
cx q[5],q[10];
cx q[7],q[10];
rz(0.59629460805556) q[10];
h q[10];
rz(1.1226334665814068) q[10];
h q[10];
cx q[7],q[10];
h q[11];
cx q[1],q[11];
rz(0.4815280003336343) q[11];
cx q[1],q[11];
cx q[11],q[3];
rz(0.49602014395666405) q[3];
h q[3];
rz(1.1226334665814068) q[3];
h q[3];
cx q[11],q[3];
h q[12];
cx q[12],q[1];
rz(0.5398855163734382) q[1];
h q[1];
rz(1.1226334665814068) q[1];
h q[1];
cx q[12],q[1];
cx q[1],q[10];
rz(11.19226345936433) q[10];
cx q[1],q[10];
h q[13];
cx q[6],q[13];
rz(0.4143337684503882) q[13];
cx q[6],q[13];
cx q[13],q[11];
rz(0.5183931616851734) q[11];
h q[11];
rz(1.1226334665814068) q[11];
h q[11];
cx q[13],q[11];
cx q[1],q[11];
rz(10.804999608230165) q[11];
cx q[1],q[11];
cx q[12],q[13];
rz(0.40874454970803065) q[13];
h q[13];
rz(1.1226334665814068) q[13];
h q[13];
cx q[12],q[13];
h q[14];
cx q[0],q[14];
rz(0.5248509967844838) q[14];
cx q[0],q[14];
h q[15];
cx q[0],q[15];
rz(0.4671771262291495) q[15];
cx q[0],q[15];
cx q[14],q[15];
rz(0.4573376916940126) q[15];
cx q[14],q[15];
h q[16];
cx q[5],q[16];
rz(0.4240881058520448) q[16];
cx q[5],q[16];
cx q[16],q[12];
rz(0.44373715047573636) q[12];
h q[12];
rz(1.1226334665814068) q[12];
h q[12];
cx q[16],q[12];
cx q[12],q[1];
rz(-1.2133616475460385) q[1];
h q[1];
rz(0.5359099936472278) q[1];
h q[1];
cx q[12],q[1];
h q[17];
cx q[17],q[6];
rz(0.5232401305933223) q[6];
h q[6];
rz(1.1226334665814068) q[6];
h q[6];
cx q[17],q[6];
cx q[17],q[7];
rz(0.5252831021578253) q[7];
h q[7];
rz(1.1226334665814068) q[7];
h q[7];
cx q[17],q[7];
cx q[17],q[8];
rz(0.44720241152398366) q[8];
h q[8];
rz(1.1226334665814068) q[8];
h q[8];
cx q[17],q[8];
h q[17];
rz(1.1226334665814068) q[17];
h q[17];
h q[18];
cx q[18],q[14];
rz(0.4988189879282343) q[14];
h q[14];
rz(1.1226334665814068) q[14];
h q[14];
cx q[18],q[14];
h q[19];
cx q[19],q[2];
rz(0.4353397548869449) q[2];
h q[2];
rz(1.1226334665814068) q[2];
h q[2];
cx q[19],q[2];
cx q[2],q[3];
rz(11.08348891674531) q[3];
cx q[2],q[3];
cx q[2],q[7];
rz(9.721456306037993) q[7];
cx q[2],q[7];
cx q[3],q[4];
rz(10.935393913050174) q[4];
cx q[3],q[4];
cx q[11],q[3];
rz(-1.6252817605563914) q[3];
h q[3];
rz(0.5359099936472278) q[3];
h q[3];
cx q[11],q[3];
cx q[4],q[8];
rz(10.219110019036577) q[8];
cx q[4],q[8];
cx q[6],q[8];
rz(4.147888486141712) q[8];
cx q[6],q[8];
cx q[6],q[13];
rz(10.174008599467953) q[13];
cx q[6],q[13];
cx q[13],q[11];
rz(-1.4151867474889555) q[11];
h q[11];
rz(0.5359099936472278) q[11];
h q[11];
cx q[13],q[11];
cx q[12],q[13];
rz(-2.444847870798622) q[13];
h q[13];
rz(0.5359099936472278) q[13];
h q[13];
cx q[12],q[13];
cx q[17],q[6];
rz(-1.369671028132181) q[6];
h q[6];
rz(0.5359099936472278) q[6];
h q[6];
cx q[17],q[6];
cx q[9],q[19];
rz(0.4784153744009382) q[19];
cx q[9],q[19];
cx q[18],q[19];
rz(0.4845800424136506) q[19];
h q[19];
rz(1.1226334665814068) q[19];
h q[19];
cx q[18],q[19];
cx q[19],q[2];
rz(-2.195104181887362) q[2];
h q[2];
rz(0.5359099936472278) q[2];
h q[2];
cx q[19],q[2];
cx q[2],q[3];
rz(11.175515937304652) q[3];
cx q[2],q[3];
h q[20];
cx q[20],q[5];
rz(0.49754946191453797) q[5];
h q[5];
rz(1.1226334665814068) q[5];
h q[5];
cx q[20],q[5];
cx q[20],q[9];
rz(0.3930510555378808) q[9];
h q[9];
rz(1.1226334665814068) q[9];
h q[9];
cx q[20],q[9];
cx q[20],q[16];
rz(0.4651539794596866) q[16];
h q[16];
rz(1.1226334665814068) q[16];
h q[16];
cx q[20],q[16];
h q[20];
rz(1.1226334665814068) q[20];
h q[20];
cx q[5],q[10];
rz(5.06047588113366) q[10];
cx q[5],q[10];
cx q[5],q[16];
rz(10.265607223700652) q[16];
cx q[5],q[16];
cx q[16],q[12];
rz(-2.1162479919281685) q[12];
h q[12];
rz(0.5359099936472278) q[12];
h q[12];
cx q[16],q[12];
cx q[20],q[5];
rz(-1.6109206189239735) q[5];
h q[5];
rz(0.5359099936472278) q[5];
h q[5];
cx q[20],q[5];
cx q[7],q[10];
rz(-0.683649071536423) q[10];
h q[10];
rz(0.5359099936472278) q[10];
h q[10];
cx q[7],q[10];
cx q[1],q[10];
rz(11.286375805745557) q[10];
cx q[1],q[10];
cx q[1],q[11];
rz(10.891687687046485) q[11];
cx q[1],q[11];
cx q[12],q[1];
rz(-1.1161676356448766) q[1];
h q[1];
rz(3.0114938608884394) q[1];
h q[1];
cx q[12],q[1];
cx q[17],q[7];
rz(-1.3504863949839345) q[7];
h q[7];
rz(0.5359099936472278) q[7];
h q[7];
cx q[17],q[7];
cx q[17],q[8];
rz(-2.0837072737269815) q[8];
h q[8];
rz(0.5359099936472278) q[8];
h q[8];
cx q[17],q[8];
h q[17];
rz(0.5359099936472278) q[17];
h q[17];
cx q[2],q[7];
rz(9.787371685904798) q[7];
cx q[2],q[7];
cx q[5],q[10];
rz(5.1574906859941505) q[10];
cx q[5],q[10];
cx q[7],q[10];
rz(-0.5762998956939986) q[10];
h q[10];
rz(3.0114938608884394) q[10];
h q[10];
cx q[7],q[10];
cx q[9],q[4];
rz(-0.6091966403771991) q[4];
h q[4];
rz(0.5359099936472278) q[4];
h q[4];
cx q[9],q[4];
cx q[3],q[4];
rz(11.024581791951341) q[4];
cx q[3],q[4];
cx q[11],q[3];
rz(-1.5359847034714362) q[3];
h q[3];
rz(3.0114938608884394) q[3];
h q[3];
cx q[11],q[3];
cx q[4],q[8];
rz(10.294565959710056) q[8];
cx q[4],q[8];
cx q[6],q[8];
rz(4.227408001997198) q[8];
cx q[6],q[8];
cx q[6],q[13];
rz(10.2485998970681) q[13];
cx q[6],q[13];
cx q[13],q[11];
rz(-1.3218619413990318) q[11];
h q[11];
rz(3.0114938608884394) q[11];
h q[11];
cx q[13],q[11];
cx q[12],q[13];
rz(-2.37126278391628) q[13];
h q[13];
rz(3.0114938608884394) q[13];
h q[13];
cx q[12],q[13];
cx q[17],q[6];
rz(-1.2754736363931585) q[6];
h q[6];
rz(3.0114938608884394) q[6];
h q[6];
cx q[17],q[6];
cx q[17],q[7];
rz(-1.255921213043969) q[7];
h q[7];
rz(3.0114938608884394) q[7];
h q[7];
cx q[17],q[7];
cx q[17],q[8];
rz(-2.003198730360493) q[8];
h q[8];
rz(3.0114938608884394) q[8];
h q[8];
cx q[17],q[8];
h q[17];
rz(3.0114938608884394) q[17];
h q[17];
cx q[9],q[19];
rz(4.492585021659693) q[19];
cx q[9],q[19];
cx q[20],q[9];
rz(-2.592218462587412) q[9];
h q[9];
rz(0.5359099936472278) q[9];
h q[9];
cx q[20],q[9];
cx q[20],q[16];
rz(-1.915132120129062) q[16];
h q[16];
rz(0.5359099936472278) q[16];
h q[16];
cx q[20],q[16];
h q[20];
rz(0.5359099936472278) q[20];
h q[20];
cx q[5],q[16];
rz(10.341954566159849) q[16];
cx q[5],q[16];
cx q[16],q[12];
rz(-2.0363632893822787) q[12];
h q[12];
rz(3.0114938608884394) q[12];
h q[12];
cx q[16],q[12];
cx q[20],q[5];
rz(-1.521348243195916) q[5];
h q[5];
rz(3.0114938608884394) q[5];
h q[5];
cx q[20],q[5];
cx q[9],q[4];
rz(-0.5004201307721079) q[4];
h q[4];
rz(3.0114938608884394) q[4];
h q[4];
cx q[9],q[4];
h q[21];
cx q[21],q[0];
rz(0.5109379934779703) q[0];
h q[0];
rz(1.1226334665814068) q[0];
h q[0];
cx q[21],q[0];
cx q[0],q[14];
rz(11.211826510881895) q[14];
cx q[0],q[14];
cx q[21],q[15];
rz(0.44989715570398836) q[15];
h q[15];
rz(1.1226334665814068) q[15];
h q[15];
cx q[21],q[15];
cx q[0],q[15];
rz(10.670236961585992) q[15];
cx q[0],q[15];
cx q[14],q[15];
rz(4.294653921015191) q[15];
cx q[14],q[15];
cx q[21],q[18];
rz(0.4827604254026907) q[18];
h q[18];
rz(1.1226334665814068) q[18];
h q[18];
cx q[21],q[18];
h q[21];
rz(1.1226334665814068) q[21];
h q[21];
cx q[18],q[14];
rz(-1.598999067362462) q[14];
h q[14];
rz(0.5359099936472278) q[14];
h q[14];
cx q[18],q[14];
cx q[18],q[19];
rz(-1.7327106426293555) q[19];
h q[19];
rz(0.5359099936472278) q[19];
h q[19];
cx q[18],q[19];
cx q[19],q[2];
rz(-2.1167312379311047) q[2];
h q[2];
rz(3.0114938608884394) q[2];
h q[2];
cx q[19],q[2];
cx q[21],q[0];
rz(-1.4851949011014414) q[0];
h q[0];
rz(0.5359099936472278) q[0];
h q[0];
cx q[21],q[0];
cx q[0],q[14];
rz(11.306313902154166) q[14];
cx q[0],q[14];
cx q[21],q[15];
rz(-2.0584021351631305) q[15];
h q[15];
rz(0.5359099936472278) q[15];
h q[15];
cx q[21],q[15];
cx q[0],q[15];
rz(10.754341494472293) q[15];
cx q[0],q[15];
cx q[14],q[15];
rz(4.376987089254161) q[15];
cx q[14],q[15];
cx q[21],q[18];
rz(-1.7497978529616507) q[18];
h q[18];
rz(0.5359099936472278) q[18];
h q[18];
cx q[21],q[18];
h q[21];
rz(0.5359099936472278) q[21];
h q[21];
cx q[18],q[14];
rz(-1.509198142575868) q[14];
h q[14];
rz(3.0114938608884394) q[14];
h q[14];
cx q[18],q[14];
cx q[21],q[0];
rz(-1.393212227158946) q[0];
h q[0];
rz(3.0114938608884394) q[0];
h q[0];
cx q[21],q[0];
cx q[21],q[15];
rz(-1.9774084648762882) q[15];
h q[15];
rz(3.0114938608884394) q[15];
h q[15];
cx q[21],q[15];
cx q[9],q[19];
rz(4.5787127435248225) q[19];
cx q[9],q[19];
cx q[18],q[19];
rz(-1.6454731135958114) q[19];
h q[19];
rz(3.0114938608884394) q[19];
h q[19];
cx q[18],q[19];
cx q[20],q[9];
rz(-2.521458629602212) q[9];
h q[9];
rz(3.0114938608884394) q[9];
h q[9];
cx q[20],q[9];
cx q[20],q[16];
rz(-1.8313918084437875) q[16];
h q[16];
rz(3.0114938608884394) q[16];
h q[16];
cx q[20],q[16];
h q[20];
rz(3.0114938608884394) q[20];
h q[20];
cx q[21],q[18];
rz(-1.662887904261447) q[18];
h q[18];
rz(3.0114938608884394) q[18];
h q[18];
cx q[21],q[18];
h q[21];
rz(3.0114938608884394) q[21];
h q[21];
