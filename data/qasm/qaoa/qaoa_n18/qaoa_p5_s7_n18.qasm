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
rz(4.4844688625964695) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(4.332854149915972) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(4.2106997069539736) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(4.81068966734346) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(4.827627414154902) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(4.813150553005229) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(4.656329172517166) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(4.422564748579246) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(4.4350155921291154) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(4.083826524157519) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(4.990247488004908) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(4.491851982239731) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(4.673228072358528) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(5.135956543648073) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(4.305913104043954) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(4.629761972308706) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(4.25276181974615) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(4.109360567113977) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(4.3221470394526) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(4.784424631219525) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(4.291377532484231) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(4.509354548491891) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(4.673321845111626) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(4.815319238904954) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(5.210586721999345) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(4.672776048670931) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(4.233174522853526) q[16];
cx q[17],q[16];
rx(2.1046398926763046) q[0];
rx(2.1046398926763046) q[1];
rx(2.1046398926763046) q[2];
rx(2.1046398926763046) q[3];
rx(2.1046398926763046) q[4];
rx(2.1046398926763046) q[5];
rx(2.1046398926763046) q[6];
rx(2.1046398926763046) q[7];
rx(2.1046398926763046) q[8];
rx(2.1046398926763046) q[9];
rx(2.1046398926763046) q[10];
rx(2.1046398926763046) q[11];
rx(2.1046398926763046) q[12];
rx(2.1046398926763046) q[13];
rx(2.1046398926763046) q[14];
rx(2.1046398926763046) q[15];
rx(2.1046398926763046) q[16];
rx(2.1046398926763046) q[17];
cx q[0],q[1];
rz(2.1899578287747152) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(2.1159178839862838) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(2.056264648144803) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(2.349265391644317) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(2.3575368173957374) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(2.3504671473005243) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(2.273884564068974) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(2.1597274038840206) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(2.1658076829855903) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(1.9943070499453608) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(2.4369511504540218) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(2.1935633216789916) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(2.2821370191844372) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(2.508107110480995) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(2.102761419724729) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(2.2609106646245123) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(2.0768053757150904) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(2.0067764145426183) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(2.1106891442846005) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(2.3364390518379663) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(2.0956630788267834) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(2.2021105729058346) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(2.282182812427977) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(2.351526209323202) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(2.5445522165461516) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(2.2819162766965806) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(2.0672400614072326) q[16];
cx q[17],q[16];
rx(5.969278373574352) q[0];
rx(5.969278373574352) q[1];
rx(5.969278373574352) q[2];
rx(5.969278373574352) q[3];
rx(5.969278373574352) q[4];
rx(5.969278373574352) q[5];
rx(5.969278373574352) q[6];
rx(5.969278373574352) q[7];
rx(5.969278373574352) q[8];
rx(5.969278373574352) q[9];
rx(5.969278373574352) q[10];
rx(5.969278373574352) q[11];
rx(5.969278373574352) q[12];
rx(5.969278373574352) q[13];
rx(5.969278373574352) q[14];
rx(5.969278373574352) q[15];
rx(5.969278373574352) q[16];
rx(5.969278373574352) q[17];
cx q[0],q[1];
rz(4.73294946411328) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(4.572933909290725) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(4.444010992648752) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(5.07724588590088) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(5.095122138842034) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(5.079843127141924) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(4.9143324074816395) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(4.667615295844669) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(4.680756030034758) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(4.310107874815718) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(5.266752852526332) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(4.740741676129851) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(4.932167660975176) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(5.420535522883303) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(4.544500083005043) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(4.886293141329433) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(4.488403731298282) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(4.337056737352237) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(4.561633526952808) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(5.049525526487855) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(4.529159107801049) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(4.7592140446760345) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(4.932266629596682) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(5.082131978080894) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(5.499300895875505) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(4.93169059103991) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(4.467731118961825) q[16];
cx q[17],q[16];
rx(4.912051739110846) q[0];
rx(4.912051739110846) q[1];
rx(4.912051739110846) q[2];
rx(4.912051739110846) q[3];
rx(4.912051739110846) q[4];
rx(4.912051739110846) q[5];
rx(4.912051739110846) q[6];
rx(4.912051739110846) q[7];
rx(4.912051739110846) q[8];
rx(4.912051739110846) q[9];
rx(4.912051739110846) q[10];
rx(4.912051739110846) q[11];
rx(4.912051739110846) q[12];
rx(4.912051739110846) q[13];
rx(4.912051739110846) q[14];
rx(4.912051739110846) q[15];
rx(4.912051739110846) q[16];
rx(4.912051739110846) q[17];
cx q[0],q[1];
rz(4.6631674684459075) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(4.505511162298625) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(4.378489068490014) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(5.002387628254451) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(5.020000316030849) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(5.004946576106055) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(4.841876125121743) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(4.59879657871674) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(4.611743567618976) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(4.246560200933541) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(5.189100528637645) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(4.6708447930948225) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(4.859448417941084) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(5.340615847162964) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(4.477496560675644) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(4.814250266288584) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(4.422227286333946) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(4.273111732921102) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(4.494377391338523) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(4.9750759742406965) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(4.4623817708288644) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(4.689044807424755) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(4.859545927382017) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(5.007201680522371) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(5.418219913667837) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(4.858978381863333) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(4.401859468324036) q[16];
cx q[17],q[16];
rx(5.845664160950254) q[0];
rx(5.845664160950254) q[1];
rx(5.845664160950254) q[2];
rx(5.845664160950254) q[3];
rx(5.845664160950254) q[4];
rx(5.845664160950254) q[5];
rx(5.845664160950254) q[6];
rx(5.845664160950254) q[7];
rx(5.845664160950254) q[8];
rx(5.845664160950254) q[9];
rx(5.845664160950254) q[10];
rx(5.845664160950254) q[11];
rx(5.845664160950254) q[12];
rx(5.845664160950254) q[13];
rx(5.845664160950254) q[14];
rx(5.845664160950254) q[15];
rx(5.845664160950254) q[16];
rx(5.845664160950254) q[17];
cx q[0],q[1];
rz(3.533329883136829) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(3.4138720807858425) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(3.3176261357475116) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(3.7903604821286825) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(3.803705796545076) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(3.792299423197097) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(3.6687392676980077) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(3.484555398878392) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(3.494365465339538) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(3.217662277846213) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(3.9318347643531966) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(3.5391470708720134) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(3.682053974440569) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(4.046639477313017) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(3.392645129416746) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(3.647806870733279) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(3.35076703261404) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(3.237780646326335) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(3.4054359081818077) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(3.7696661613826055) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(3.3811924979158756) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(3.5529374086500605) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(3.6821278583454182) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(3.7940081389740192) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(4.105440875523574) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(3.681697823277029) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(3.335334127723603) q[16];
cx q[17],q[16];
rx(3.888021620653176) q[0];
rx(3.888021620653176) q[1];
rx(3.888021620653176) q[2];
rx(3.888021620653176) q[3];
rx(3.888021620653176) q[4];
rx(3.888021620653176) q[5];
rx(3.888021620653176) q[6];
rx(3.888021620653176) q[7];
rx(3.888021620653176) q[8];
rx(3.888021620653176) q[9];
rx(3.888021620653176) q[10];
rx(3.888021620653176) q[11];
rx(3.888021620653176) q[12];
rx(3.888021620653176) q[13];
rx(3.888021620653176) q[14];
rx(3.888021620653176) q[15];
rx(3.888021620653176) q[16];
rx(3.888021620653176) q[17];
