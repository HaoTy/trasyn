OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.7987127728490875) q[1];
cx q[0],q[1];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.7515710591851368) q[3];
cx q[2],q[3];
h q[4];
h q[5];
h q[6];
cx q[1],q[6];
rz(0.8657157030694046) q[6];
cx q[1],q[6];
h q[7];
h q[8];
h q[9];
cx q[0],q[9];
rz(0.5384052206277351) q[9];
cx q[0],q[9];
cx q[4],q[9];
rz(0.8583189661576922) q[9];
cx q[4],q[9];
cx q[7],q[9];
rz(-2.398429039008982) q[9];
h q[9];
rz(2.5377436668568585) q[9];
h q[9];
rz(3*pi) q[9];
cx q[7],q[9];
h q[10];
cx q[3],q[10];
rz(0.7771358552592499) q[10];
cx q[3],q[10];
cx q[4],q[10];
rz(0.8302882568373483) q[10];
cx q[4],q[10];
h q[11];
cx q[5],q[11];
rz(0.8943575675050471) q[11];
cx q[5],q[11];
cx q[8],q[11];
rz(0.8273761642404382) q[11];
cx q[8],q[11];
h q[12];
cx q[5],q[12];
rz(0.8350654292961398) q[12];
cx q[5],q[12];
cx q[6],q[12];
rz(0.6914465979904486) q[12];
cx q[6],q[12];
h q[13];
cx q[13],q[3];
rz(-2.3777273105737615) q[3];
h q[3];
rz(2.5377436668568585) q[3];
h q[3];
rz(3*pi) q[3];
cx q[13],q[3];
cx q[8],q[13];
rz(0.7356522499899073) q[13];
cx q[8],q[13];
h q[14];
cx q[14],q[5];
rz(-2.318438878318536) q[5];
h q[5];
rz(2.5377436668568585) q[5];
h q[5];
rz(3*pi) q[5];
cx q[14],q[5];
cx q[14],q[8];
rz(-2.3729120832167734) q[8];
h q[8];
rz(2.5377436668568585) q[8];
h q[8];
rz(3*pi) q[8];
cx q[14],q[8];
h q[15];
cx q[15],q[11];
rz(-2.452589369716633) q[11];
h q[11];
rz(2.5377436668568585) q[11];
h q[11];
rz(3*pi) q[11];
cx q[15],q[11];
cx q[15],q[13];
rz(-2.3641049756691084) q[13];
h q[13];
rz(2.5377436668568585) q[13];
h q[13];
rz(3*pi) q[13];
cx q[15],q[13];
cx q[5],q[11];
rz(8.61381618137321) q[11];
cx q[5],q[11];
cx q[8],q[11];
rz(2.156082201361568) q[11];
cx q[8],q[11];
h q[16];
cx q[2],q[16];
rz(0.7473215389858926) q[16];
cx q[2],q[16];
cx q[7],q[16];
rz(0.7008869343271998) q[16];
cx q[7],q[16];
cx q[16],q[10];
rz(-2.348442973241485) q[10];
h q[10];
rz(2.5377436668568585) q[10];
h q[10];
rz(3*pi) q[10];
cx q[16],q[10];
h q[17];
cx q[17],q[1];
rz(-2.2729857883405518) q[1];
h q[1];
rz(2.5377436668568585) q[1];
h q[1];
rz(3*pi) q[1];
cx q[17],q[1];
cx q[17],q[2];
rz(-2.3479314812324903) q[2];
h q[2];
rz(2.5377436668568585) q[2];
h q[2];
rz(3*pi) q[2];
cx q[17],q[2];
cx q[17],q[15];
rz(-2.336266093416156) q[15];
h q[15];
rz(2.5377436668568585) q[15];
h q[15];
rz(3*pi) q[15];
cx q[17],q[15];
h q[17];
rz(3.7454416403227277) q[17];
h q[17];
cx q[15],q[11];
rz(-4.487692783652056) q[11];
h q[11];
rz(1.5790034962595207) q[11];
h q[11];
cx q[15],q[11];
cx q[2],q[3];
rz(8.241724909604763) q[3];
cx q[2],q[3];
cx q[2],q[16];
rz(-pi) q[16];
h q[16];
rz(2.5377436668568585) q[16];
h q[16];
rz(5.089058314968663) q[16];
cx q[2],q[16];
cx q[3],q[10];
rz(8.308344911701916) q[10];
cx q[3],q[10];
cx q[13],q[3];
rz(-4.2926076938810676) q[3];
h q[3];
rz(1.5790034962595207) q[3];
h q[3];
cx q[13],q[3];
cx q[8],q[13];
rz(1.9170563416592405) q[13];
cx q[8],q[13];
cx q[15],q[13];
rz(-4.257108878350319) q[13];
h q[13];
rz(1.5790034962595207) q[13];
h q[13];
cx q[15],q[13];
h q[18];
cx q[18],q[4];
rz(-2.3500019720155105) q[4];
h q[4];
rz(2.5377436668568585) q[4];
h q[4];
rz(3*pi) q[4];
cx q[18],q[4];
cx q[18],q[7];
rz(-2.3515212986258547) q[7];
h q[7];
rz(2.5377436668568585) q[7];
h q[7];
rz(3*pi) q[7];
cx q[18],q[7];
cx q[18],q[12];
rz(-2.118658796499058) q[12];
h q[12];
rz(2.5377436668568585) q[12];
h q[12];
rz(3*pi) q[12];
cx q[18],q[12];
h q[18];
rz(3.7454416403227277) q[18];
h q[18];
cx q[5],q[12];
rz(8.459305174894968) q[12];
cx q[5],q[12];
h q[19];
cx q[19],q[0];
rz(-2.4622683229896527) q[0];
h q[0];
rz(2.5377436668568585) q[0];
h q[0];
rz(3*pi) q[0];
cx q[19],q[0];
cx q[0],q[1];
rz(8.364572792061999) q[1];
cx q[0],q[1];
cx q[0],q[9];
rz(7.686230214064602) q[9];
cx q[0],q[9];
cx q[19],q[6];
rz(-2.5486328228508257) q[6];
h q[6];
rz(2.5377436668568585) q[6];
h q[6];
rz(3*pi) q[6];
cx q[19],q[6];
cx q[1],q[6];
rz(8.539177562577379) q[6];
cx q[1],q[6];
cx q[17],q[1];
rz(-4.019658893092418) q[1];
h q[1];
rz(1.5790034962595207) q[1];
h q[1];
cx q[17],q[1];
cx q[17],q[2];
rz(-4.214961926340133) q[2];
h q[2];
rz(1.5790034962595207) q[2];
h q[2];
cx q[17],q[2];
cx q[17],q[15];
rz(-4.184562772811377) q[15];
h q[15];
rz(1.5790034962595207) q[15];
h q[15];
cx q[17],q[15];
h q[17];
rz(1.5790034962595207) q[17];
h q[17];
cx q[19],q[14];
rz(-2.3797453168972194) q[14];
h q[14];
rz(2.5377436668568585) q[14];
h q[14];
rz(3*pi) q[14];
cx q[19],q[14];
h q[19];
rz(3.7454416403227277) q[19];
h q[19];
cx q[14],q[5];
rz(-4.138106344849002) q[5];
h q[5];
rz(1.5790034962595207) q[5];
h q[5];
cx q[14],q[5];
cx q[14],q[8];
rz(-4.280059561062372) q[8];
h q[8];
rz(1.5790034962595207) q[8];
h q[8];
cx q[14],q[8];
cx q[19],q[0];
rz(-4.512915433012431) q[0];
h q[0];
rz(1.5790034962595207) q[0];
h q[0];
cx q[19],q[0];
cx q[0],q[1];
rz(11.913918074886826) q[1];
cx q[0],q[1];
cx q[2],q[3];
rz(11.581580342379283) q[3];
cx q[2],q[3];
cx q[4],q[9];
cx q[5],q[11];
rz(0.021820237472084614) q[11];
cx q[5],q[11];
cx q[6],q[12];
rz(1.8018596226878703) q[12];
cx q[6],q[12];
cx q[19],q[6];
rz(1.5452102591719434) q[6];
h q[6];
rz(1.5790034962595207) q[6];
h q[6];
cx q[19],q[6];
cx q[1],q[6];
rz(12.386272602758169) q[6];
cx q[1],q[6];
cx q[17],q[1];
rz(-0.1597160142470715) q[1];
h q[1];
rz(0.6871102720737552) q[1];
h q[1];
cx q[17],q[1];
cx q[19],q[14];
rz(-4.297866471815813) q[14];
h q[14];
rz(1.5790034962595207) q[14];
h q[14];
cx q[19],q[14];
h q[19];
rz(1.5790034962595207) q[19];
h q[19];
cx q[8],q[11];
rz(5.832802776635708) q[11];
cx q[8],q[11];
cx q[15],q[11];
rz(-1.4258780254597712) q[11];
h q[11];
rz(0.6871102720737552) q[11];
h q[11];
cx q[15],q[11];
rz(2.236716896144315) q[9];
cx q[4],q[9];
cx q[4],q[10];
rz(2.1636709031979064) q[10];
cx q[4],q[10];
cx q[18],q[4];
rz(-4.220357474965169) q[4];
h q[4];
rz(1.5790034962595207) q[4];
h q[4];
cx q[18],q[4];
cx q[7],q[9];
rz(-4.346554894858962) q[9];
h q[9];
rz(1.5790034962595207) q[9];
h q[9];
cx q[7],q[9];
cx q[0],q[9];
rz(10.078812497764204) q[9];
cx q[0],q[9];
cx q[19],q[0];
rz(-1.494112315938306) q[0];
h q[0];
rz(0.6871102720737552) q[0];
h q[0];
cx q[19],q[0];
cx q[0],q[1];
rz(12.314835619212172) q[1];
cx q[0],q[1];
cx q[4],q[9];
cx q[7],q[16];
rz(1.8264604536402846) q[16];
cx q[7],q[16];
cx q[16],q[10];
rz(-4.216294837371896) q[10];
h q[10];
rz(1.5790034962595207) q[10];
h q[10];
cx q[16],q[10];
cx q[18],q[7];
rz(-4.224316729780813) q[7];
h q[7];
rz(1.5790034962595207) q[7];
h q[7];
cx q[18],q[7];
cx q[18],q[12];
rz(-3.6174939593169997) q[12];
h q[12];
rz(1.5790034962595207) q[12];
h q[12];
cx q[18],q[12];
h q[18];
rz(1.5790034962595207) q[18];
h q[18];
cx q[2],q[16];
h q[16];
rz(1.5790034962595207) q[16];
h q[16];
rz(5.26843694081795) q[16];
cx q[2],q[16];
cx q[17],q[2];
rz(-0.6880651068249275) q[2];
h q[2];
rz(0.6871102720737552) q[2];
h q[2];
cx q[17],q[2];
cx q[3],q[10];
rz(11.761806000272273) q[10];
cx q[3],q[10];
cx q[13],q[3];
rz(-0.8981185306789738) q[3];
h q[3];
rz(0.6871102720737552) q[3];
h q[3];
cx q[13],q[3];
cx q[2],q[3];
rz(11.958834887038272) q[3];
cx q[2],q[3];
cx q[5],q[12];
rz(12.170195551356201) q[12];
cx q[5],q[12];
cx q[14],q[5];
rz(-0.4801493562852084) q[5];
h q[5];
rz(0.6871102720737552) q[5];
h q[5];
cx q[14],q[5];
cx q[5],q[11];
rz(0.47074712571237143) q[11];
cx q[5],q[11];
cx q[6],q[12];
rz(4.874532058047032) q[12];
cx q[6],q[12];
cx q[19],q[6];
rz(-2.1029612487033216) q[6];
h q[6];
rz(0.6871102720737552) q[6];
h q[6];
cx q[19],q[6];
cx q[1],q[6];
rz(0.2544519612514442) q[6];
cx q[1],q[6];
cx q[8],q[13];
rz(5.186171262642857) q[13];
cx q[8],q[13];
cx q[14],q[8];
rz(-0.8641723370226009) q[8];
h q[8];
rz(0.6871102720737552) q[8];
h q[8];
cx q[14],q[8];
cx q[15],q[13];
rz(-0.8020843490210261) q[13];
h q[13];
rz(0.6871102720737552) q[13];
h q[13];
cx q[15],q[13];
cx q[17],q[15];
rz(-0.6058269310272086) q[15];
h q[15];
rz(0.6871102720737552) q[15];
h q[15];
cx q[17],q[15];
h q[17];
rz(0.6871102720737552) q[17];
h q[17];
cx q[17],q[1];
rz(0.2762851907388644) q[1];
h q[1];
rz(2.1096167557801717) q[1];
h q[1];
cx q[17],q[1];
cx q[19],q[14];
rz(-0.9123449894476261) q[14];
h q[14];
rz(0.6871102720737552) q[14];
h q[14];
cx q[19],q[14];
h q[19];
rz(0.6871102720737552) q[19];
h q[19];
cx q[8],q[11];
rz(6.248108041903168) q[11];
cx q[8],q[11];
cx q[15],q[11];
rz(-1.0800296629537183) q[11];
h q[11];
rz(2.1096167557801717) q[11];
h q[11];
cx q[15],q[11];
rz(6.050942081030025) q[9];
cx q[4],q[9];
cx q[4],q[10];
rz(5.853332328391251) q[10];
cx q[4],q[10];
cx q[18],q[4];
rz(-0.7026615683984732) q[4];
h q[4];
rz(0.6871102720737552) q[4];
h q[4];
cx q[18],q[4];
cx q[7],q[9];
rz(-1.044060732445161) q[9];
h q[9];
rz(0.6871102720737552) q[9];
h q[9];
cx q[7],q[9];
cx q[0],q[9];
rz(10.349067470820462) q[9];
cx q[0],q[9];
cx q[19],q[0];
rz(-1.1531223484770914) q[0];
h q[0];
rz(2.1096167557801717) q[0];
h q[0];
cx q[19],q[0];
cx q[4],q[9];
cx q[7],q[16];
rz(4.941084156569145) q[16];
cx q[7],q[16];
cx q[16],q[10];
rz(-0.6916710023525461) q[10];
h q[10];
rz(0.6871102720737552) q[10];
h q[10];
cx q[16],q[10];
cx q[18],q[7];
rz(-0.7133724552410579) q[7];
h q[7];
rz(0.6871102720737552) q[7];
h q[7];
cx q[18],q[7];
cx q[18],q[12];
rz(0.9282521257938949) q[12];
h q[12];
rz(0.6871102720737552) q[12];
h q[12];
cx q[18],q[12];
h q[18];
rz(0.6871102720737552) q[18];
h q[18];
cx q[2],q[16];
h q[16];
rz(0.6871102720737552) q[16];
h q[16];
rz(5.643558419297514) q[16];
cx q[2],q[16];
cx q[17],q[2];
rz(-0.2896832365653106) q[2];
h q[2];
rz(2.1096167557801717) q[2];
h q[2];
cx q[17],q[2];
cx q[3],q[10];
rz(12.151892911723962) q[10];
cx q[3],q[10];
cx q[13],q[3];
rz(-0.5146928137910232) q[3];
h q[3];
rz(2.1096167557801717) q[3];
h q[3];
cx q[13],q[3];
cx q[5],q[12];
rz(0.022989864171236718) q[12];
cx q[5],q[12];
cx q[14],q[5];
rz(-0.06696353855164894) q[5];
h q[5];
rz(2.1096167557801717) q[5];
h q[5];
cx q[14],q[5];
cx q[6],q[12];
rz(5.221606853294888) q[12];
cx q[6],q[12];
cx q[19],q[6];
rz(-1.8053223385661523) q[6];
h q[6];
rz(2.1096167557801717) q[6];
h q[6];
cx q[19],q[6];
cx q[8],q[13];
rz(5.555435287921058) q[13];
cx q[8],q[13];
cx q[14],q[8];
rz(-0.47832959464944924) q[8];
h q[8];
rz(2.1096167557801717) q[8];
h q[8];
cx q[14],q[8];
cx q[15],q[13];
rz(-0.4118208385687261) q[13];
h q[13];
rz(2.1096167557801717) q[13];
h q[13];
cx q[15],q[13];
cx q[17],q[15];
rz(-0.20158956578127185) q[15];
h q[15];
rz(2.1096167557801717) q[15];
h q[15];
cx q[17],q[15];
h q[17];
rz(2.1096167557801717) q[17];
h q[17];
cx q[19],q[14];
rz(-0.5299322201011902) q[14];
h q[14];
rz(2.1096167557801717) q[14];
h q[14];
cx q[19],q[14];
h q[19];
rz(2.1096167557801717) q[19];
h q[19];
rz(6.481779227821002) q[9];
cx q[4],q[9];
cx q[4],q[10];
rz(6.270099331911171) q[10];
cx q[4],q[10];
cx q[18],q[4];
rz(-0.30531899049587796) q[4];
h q[4];
rz(2.1096167557801717) q[4];
h q[4];
cx q[18],q[4];
cx q[7],q[9];
rz(-0.6710263432155026) q[9];
h q[9];
rz(2.1096167557801717) q[9];
h q[9];
cx q[7],q[9];
cx q[7],q[16];
rz(5.2928975719948586) q[16];
cx q[7],q[16];
cx q[16],q[10];
rz(-0.2935458778558111) q[10];
h q[10];
rz(2.1096167557801717) q[10];
h q[10];
cx q[16],q[10];
h q[16];
rz(2.1096167557801717) q[16];
h q[16];
cx q[18],q[7];
rz(-0.3167925103078515) q[7];
h q[7];
rz(2.1096167557801717) q[7];
h q[7];
cx q[18],q[7];
cx q[18],q[12];
rz(1.4417184730630899) q[12];
h q[12];
rz(2.1096167557801717) q[12];
h q[12];
cx q[18],q[12];
h q[18];
rz(2.1096167557801717) q[18];
h q[18];
