OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
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
cx q[0],q[4];
rz(3.8176987512047242) q[4];
cx q[0],q[4];
cx q[0],q[9];
rz(3.8318264356188902) q[9];
cx q[0],q[9];
cx q[11],q[0];
rz(3.9695940695177314) q[0];
cx q[11],q[0];
cx q[1],q[2];
rz(3.750965024243086) q[2];
cx q[1],q[2];
cx q[1],q[6];
rz(3.0333885243861154) q[6];
cx q[1],q[6];
cx q[9],q[1];
rz(3.445102608762614) q[1];
cx q[9],q[1];
cx q[2],q[8];
rz(3.5614407676610456) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(3.6272015866123564) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(3.7068685395441854) q[5];
cx q[3],q[5];
cx q[3],q[6];
rz(3.536989368501935) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(3.32702640383632) q[3];
cx q[7],q[3];
cx q[4],q[6];
rz(3.9445613611043577) q[6];
cx q[4],q[6];
cx q[11],q[4];
rz(3.89808415061905) q[4];
cx q[11],q[4];
cx q[5],q[7];
rz(3.9542727569245417) q[7];
cx q[5],q[7];
cx q[8],q[5];
rz(3.781508579816001) q[5];
cx q[8],q[5];
cx q[10],q[7];
rz(2.799765289083818) q[7];
cx q[10],q[7];
cx q[9],q[8];
rz(3.8377826720851305) q[8];
cx q[9],q[8];
cx q[11],q[10];
rz(3.3054624036702687) q[10];
cx q[11],q[10];
rx(1.6143471464298924) q[0];
rx(1.6143471464298924) q[1];
rx(1.6143471464298924) q[2];
rx(1.6143471464298924) q[3];
rx(1.6143471464298924) q[4];
rx(1.6143471464298924) q[5];
rx(1.6143471464298924) q[6];
rx(1.6143471464298924) q[7];
rx(1.6143471464298924) q[8];
rx(1.6143471464298924) q[9];
rx(1.6143471464298924) q[10];
rx(1.6143471464298924) q[11];
cx q[0],q[4];
rz(4.859471195582893) q[4];
cx q[0],q[4];
cx q[0],q[9];
rz(4.8774540381131635) q[9];
cx q[0],q[9];
cx q[11],q[0];
rz(5.052815660976614) q[0];
cx q[11],q[0];
cx q[1],q[2];
rz(4.774527190024141) q[2];
cx q[1],q[2];
cx q[1],q[6];
rz(3.861138638719049) q[6];
cx q[1],q[6];
cx q[9],q[1];
rz(4.385201134014685) q[1];
cx q[9],q[1];
cx q[2],q[8];
rz(4.533285613424086) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(4.616991224138131) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(4.718397670335572) q[5];
cx q[3],q[5];
cx q[3],q[6];
rz(4.502161924089536) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(4.234904331119485) q[3];
cx q[7],q[3];
cx q[4],q[6];
rz(5.020952034899824) q[6];
cx q[4],q[6];
cx q[11],q[4];
rz(4.961792137714871) q[4];
cx q[11],q[4];
cx q[5],q[7];
rz(5.0333134733821545) q[7];
cx q[5],q[7];
cx q[8],q[5];
rz(4.8134054615143755) q[5];
cx q[8],q[5];
cx q[10],q[7];
rz(3.5637643678413) q[7];
cx q[10],q[7];
cx q[9],q[8];
rz(4.885035610528389) q[8];
cx q[9],q[8];
cx q[11],q[10];
rz(4.207455953314557) q[10];
cx q[11],q[10];
rx(4.963176739708052) q[0];
rx(4.963176739708052) q[1];
rx(4.963176739708052) q[2];
rx(4.963176739708052) q[3];
rx(4.963176739708052) q[4];
rx(4.963176739708052) q[5];
rx(4.963176739708052) q[6];
rx(4.963176739708052) q[7];
rx(4.963176739708052) q[8];
rx(4.963176739708052) q[9];
rx(4.963176739708052) q[10];
rx(4.963176739708052) q[11];
cx q[0],q[4];
rz(3.3086596399807804) q[4];
cx q[0],q[4];
cx q[0],q[9];
rz(3.320903586471526) q[9];
cx q[0],q[9];
cx q[11],q[0];
rz(3.440301747427233) q[0];
cx q[11],q[0];
cx q[1],q[2];
rz(3.250823963723246) q[2];
cx q[1],q[2];
cx q[1],q[6];
rz(2.6289267008954185) q[6];
cx q[1],q[6];
cx q[9],q[1];
rz(2.985744214000216) q[1];
cx q[9],q[1];
cx q[2],q[8];
rz(3.086570234077218) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(3.143562726606291) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(3.2126071559821314) q[5];
cx q[3],q[5];
cx q[3],q[6];
rz(3.0653791022433405) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(2.8834118931070485) q[3];
cx q[7],q[3];
cx q[4],q[6];
rz(3.4186068161599086) q[6];
cx q[4],q[6];
cx q[11],q[4];
rz(3.37832672060153) q[4];
cx q[11],q[4];
cx q[5],q[7];
rz(3.427023327124264) q[7];
cx q[5],q[7];
cx q[8],q[5];
rz(3.277294944324773) q[5];
cx q[8],q[5];
cx q[10],q[7];
rz(2.4264540020313397) q[7];
cx q[10],q[7];
cx q[9],q[8];
rz(3.326065638400273) q[8];
cx q[9],q[8];
cx q[11],q[10];
rz(2.8647231641958326) q[10];
cx q[11],q[10];
rx(4.6188641108422095) q[0];
rx(4.6188641108422095) q[1];
rx(4.6188641108422095) q[2];
rx(4.6188641108422095) q[3];
rx(4.6188641108422095) q[4];
rx(4.6188641108422095) q[5];
rx(4.6188641108422095) q[6];
rx(4.6188641108422095) q[7];
rx(4.6188641108422095) q[8];
rx(4.6188641108422095) q[9];
rx(4.6188641108422095) q[10];
rx(4.6188641108422095) q[11];
cx q[0],q[4];
rz(3.5624042591033795) q[4];
cx q[0],q[4];
cx q[0],q[9];
rz(3.575587206844453) q[9];
cx q[0],q[9];
cx q[11],q[0];
rz(3.7041421394757807) q[0];
cx q[11],q[0];
cx q[1],q[2];
rz(3.500133103455239) q[2];
cx q[1],q[2];
cx q[1],q[6];
rz(2.830541879549399) q[6];
cx q[1],q[6];
cx q[9],q[1];
rz(3.2147241064086693) q[1];
cx q[9],q[1];
cx q[2],q[8];
rz(3.323282580967521) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(3.3846458882322317) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(3.45898540817067) q[5];
cx q[3],q[5];
cx q[3],q[6];
rz(3.3004662787441035) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(3.1045438112254895) q[3];
cx q[7],q[3];
cx q[4],q[6];
rz(3.680783400905705) q[6];
cx q[4],q[6];
cx q[11],q[4];
rz(3.637414182071491) q[4];
cx q[11],q[4];
cx q[5],q[7];
rz(3.6898453830279836) q[7];
cx q[5],q[7];
cx q[8],q[5];
rz(3.528634171651565) q[5];
cx q[8],q[5];
cx q[10],q[7];
rz(2.6125413345342157) q[7];
cx q[10],q[7];
cx q[9],q[8];
rz(3.5811451420138707) q[8];
cx q[9],q[8];
cx q[11],q[10];
rz(3.0844218238605605) q[10];
cx q[11],q[10];
rx(5.907650080457982) q[0];
rx(5.907650080457982) q[1];
rx(5.907650080457982) q[2];
rx(5.907650080457982) q[3];
rx(5.907650080457982) q[4];
rx(5.907650080457982) q[5];
rx(5.907650080457982) q[6];
rx(5.907650080457982) q[7];
rx(5.907650080457982) q[8];
rx(5.907650080457982) q[9];
rx(5.907650080457982) q[10];
rx(5.907650080457982) q[11];
