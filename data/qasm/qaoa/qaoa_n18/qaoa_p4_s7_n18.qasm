OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
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
cx q[0],q[1];
rz(0.930896597155224) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(1.037293960105141) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(1.0677228729705395) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(0.9624410090380712) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(0.9165350684823217) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(0.731100448145714) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(0.9158962625794983) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(0.8264343805392598) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(0.9344959432896562) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(0.9106702978386688) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(0.9482804743260601) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(0.9469799709995919) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(0.8422416539291316) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(0.9338344522614729) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(0.8804344110119888) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(0.9816142140112751) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(0.9434017682201422) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(1.0066059864026666) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(0.8679863544366182) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(1.0216775477139728) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(1.009216537689997) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(0.7382845900354291) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(0.8687112409200861) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(0.864826857048047) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(1.0254028112025027) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(1.0104487468875152) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(0.6879187407375517) q[16];
cx q[17],q[16];
rx(4.990018832475265) q[0];
rx(4.990018832475265) q[1];
rx(4.990018832475265) q[2];
rx(4.990018832475265) q[3];
rx(4.990018832475265) q[4];
rx(4.990018832475265) q[5];
rx(4.990018832475265) q[6];
rx(4.990018832475265) q[7];
rx(4.990018832475265) q[8];
rx(4.990018832475265) q[9];
rx(4.990018832475265) q[10];
rx(4.990018832475265) q[11];
rx(4.990018832475265) q[12];
rx(4.990018832475265) q[13];
rx(4.990018832475265) q[14];
rx(4.990018832475265) q[15];
rx(4.990018832475265) q[16];
rx(4.990018832475265) q[17];
cx q[0],q[1];
rz(3.9207319418924973) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(4.368854258297645) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(4.497014153814524) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(4.053590074186937) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(3.860244338463366) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(3.0792344590543705) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(3.85755382835137) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(3.4807600366790448) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(3.935891597005989) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(3.835543213048672) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(3.9939490131611213) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(3.988471578880876) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(3.5473367992145333) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(3.9331055421300776) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(3.7081963007975323) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(4.134343401029701) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(3.9734009749332095) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(4.239603255452713) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(3.655767821438291) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(4.30308136035503) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(4.250598323916239) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(3.109492486281209) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(3.65882088404338) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(3.642460712604127) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(4.318771351698944) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(4.255788118330105) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(2.8973625948129254) q[16];
cx q[17],q[16];
rx(1.3512080998548481) q[0];
rx(1.3512080998548481) q[1];
rx(1.3512080998548481) q[2];
rx(1.3512080998548481) q[3];
rx(1.3512080998548481) q[4];
rx(1.3512080998548481) q[5];
rx(1.3512080998548481) q[6];
rx(1.3512080998548481) q[7];
rx(1.3512080998548481) q[8];
rx(1.3512080998548481) q[9];
rx(1.3512080998548481) q[10];
rx(1.3512080998548481) q[11];
rx(1.3512080998548481) q[12];
rx(1.3512080998548481) q[13];
rx(1.3512080998548481) q[14];
rx(1.3512080998548481) q[15];
rx(1.3512080998548481) q[16];
rx(1.3512080998548481) q[17];
cx q[0],q[1];
rz(4.874628209940325) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(5.431776649932359) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(5.591117265748095) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(5.039810122195682) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(4.799424247415664) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(3.8283982127789993) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(4.796079148417456) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(4.32761313915243) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(4.893476140266703) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(4.768713450414423) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(4.9656586619948895) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(4.958848592840382) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(4.410387725528601) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(4.890012250897616) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(4.6103836129992795) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(5.140210366565427) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(4.940111379423881) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(5.271079465332979) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(4.5451995228149675) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(5.350001504752237) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(5.284749583998796) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(3.866017880556107) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(4.54899538168062) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(4.528654854860838) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(5.3695087997043744) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(5.291202031814294) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(3.602277750282435) q[16];
cx q[17],q[16];
rx(4.87554196181674) q[0];
rx(4.87554196181674) q[1];
rx(4.87554196181674) q[2];
rx(4.87554196181674) q[3];
rx(4.87554196181674) q[4];
rx(4.87554196181674) q[5];
rx(4.87554196181674) q[6];
rx(4.87554196181674) q[7];
rx(4.87554196181674) q[8];
rx(4.87554196181674) q[9];
rx(4.87554196181674) q[10];
rx(4.87554196181674) q[11];
rx(4.87554196181674) q[12];
rx(4.87554196181674) q[13];
rx(4.87554196181674) q[14];
rx(4.87554196181674) q[15];
rx(4.87554196181674) q[16];
rx(4.87554196181674) q[17];
cx q[0],q[1];
rz(3.3879564725538116) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(3.775184909708278) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(3.88592957523842) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(3.502760946797233) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(3.33568833216978) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(2.6608073366598015) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(3.333363426697153) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(3.0077709138108566) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(3.4010561316033523) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(3.3143437620822667) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(3.4512243149317916) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(3.446491191321768) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(3.0653007773406014) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(3.398648664632994) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(3.2043016061748326) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(3.5725409675719093) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(3.4334684825657225) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(3.663497403864359) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(3.158997461789599) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(3.718349676235075) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(3.672998388353799) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(2.686953699305649) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(3.161635653680124) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(3.14749861255077) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(3.731907605855295) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(3.6774829585402506) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(2.5036494465606354) q[16];
cx q[17],q[16];
rx(3.1900337157969942) q[0];
rx(3.1900337157969942) q[1];
rx(3.1900337157969942) q[2];
rx(3.1900337157969942) q[3];
rx(3.1900337157969942) q[4];
rx(3.1900337157969942) q[5];
rx(3.1900337157969942) q[6];
rx(3.1900337157969942) q[7];
rx(3.1900337157969942) q[8];
rx(3.1900337157969942) q[9];
rx(3.1900337157969942) q[10];
rx(3.1900337157969942) q[11];
rx(3.1900337157969942) q[12];
rx(3.1900337157969942) q[13];
rx(3.1900337157969942) q[14];
rx(3.1900337157969942) q[15];
rx(3.1900337157969942) q[16];
rx(3.1900337157969942) q[17];
