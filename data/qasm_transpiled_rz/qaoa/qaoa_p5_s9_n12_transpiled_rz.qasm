OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(4.955207975337263) q[1];
cx q[0],q[1];
h q[2];
h q[3];
cx q[2],q[3];
rz(5.360552058599617) q[3];
cx q[2],q[3];
h q[4];
h q[5];
cx q[1],q[5];
rz(6.680933271269434) q[5];
cx q[1],q[5];
h q[6];
cx q[2],q[6];
rz(5.9154761153697475) q[6];
cx q[2],q[6];
h q[7];
cx q[0],q[7];
rz(6.441699730546957) q[7];
cx q[0],q[7];
cx q[7],q[2];
rz(-2.533082500988827) q[2];
h q[2];
rz(1.3292342602863663) q[2];
h q[2];
rz(3*pi) q[2];
cx q[7],q[2];
h q[8];
cx q[3],q[8];
rz(6.02711936921718) q[8];
cx q[3],q[8];
cx q[4],q[8];
rz(4.965847200686904) q[8];
cx q[4],q[8];
cx q[5],q[8];
rz(-4.01163201797426) q[8];
h q[8];
rz(1.3292342602863663) q[8];
h q[8];
rz(3*pi) q[8];
cx q[5],q[8];
h q[9];
cx q[9],q[0];
rz(-4.015253427898923) q[0];
h q[0];
rz(1.3292342602863663) q[0];
h q[0];
rz(3*pi) q[0];
cx q[9],q[0];
cx q[4],q[9];
rz(6.553511541268349) q[9];
cx q[4],q[9];
cx q[9],q[7];
rz(-4.030686767715337) q[7];
h q[7];
rz(1.3292342602863663) q[7];
h q[7];
rz(3*pi) q[7];
cx q[9],q[7];
h q[9];
rz(4.95395104689322) q[9];
h q[9];
h q[10];
cx q[10],q[1];
rz(-2.9609622504802533) q[1];
h q[1];
rz(1.3292342602863663) q[1];
h q[1];
rz(3*pi) q[1];
cx q[10],q[1];
cx q[0],q[1];
rz(7.692147701025231) q[1];
cx q[0],q[1];
cx q[0],q[7];
rz(8.114816334163585) q[7];
cx q[0],q[7];
cx q[10],q[4];
rz(-3.209553149094772) q[4];
h q[4];
rz(1.3292342602863663) q[4];
h q[4];
rz(3*pi) q[4];
cx q[10],q[4];
cx q[6],q[10];
rz(-3.366380830648034) q[10];
h q[10];
rz(1.3292342602863663) q[10];
h q[10];
rz(3*pi) q[10];
cx q[6],q[10];
cx q[9],q[0];
rz(-1.603450021971764) q[0];
h q[0];
rz(1.0660196368850254) q[0];
h q[0];
rz(3*pi) q[0];
cx q[9],q[0];
h q[11];
cx q[11],q[3];
rz(-3.192784176236947) q[3];
h q[3];
rz(1.3292342602863663) q[3];
h q[3];
rz(3*pi) q[3];
cx q[11],q[3];
cx q[11],q[5];
rz(-3.8350403923043817) q[5];
h q[5];
rz(1.3292342602863663) q[5];
h q[5];
rz(3*pi) q[5];
cx q[11],q[5];
cx q[1],q[5];
rz(8.18283992952066) q[5];
cx q[1],q[5];
cx q[10],q[1];
rz(-1.3036731753611939) q[1];
h q[1];
rz(1.0660196368850254) q[1];
h q[1];
rz(3*pi) q[1];
cx q[10],q[1];
cx q[0],q[1];
rz(9.605040689041196) q[1];
cx q[0],q[1];
cx q[11],q[6];
rz(-3.142188948537884) q[6];
h q[6];
rz(1.3292342602863663) q[6];
h q[6];
rz(3*pi) q[6];
cx q[11],q[6];
h q[11];
rz(4.95395104689322) q[11];
h q[11];
cx q[2],q[3];
rz(7.80740311971341) q[3];
cx q[2],q[3];
cx q[2],q[6];
rz(7.965190064544149) q[6];
cx q[2],q[6];
cx q[3],q[8];
rz(7.996934674857279) q[8];
cx q[3],q[8];
cx q[11],q[3];
rz(-1.3695893542755262) q[3];
h q[3];
rz(1.0660196368850254) q[3];
h q[3];
rz(3*pi) q[3];
cx q[11],q[3];
cx q[4],q[8];
cx q[7],q[2];
rz(-1.182009971894848) q[2];
h q[2];
rz(1.0660196368850254) q[2];
h q[2];
rz(3*pi) q[2];
cx q[7],q[2];
cx q[2],q[3];
rz(9.876773869806264) q[3];
cx q[2],q[3];
rz(1.4119875480857702) q[8];
cx q[4],q[8];
cx q[4],q[9];
cx q[5],q[8];
rz(-1.6024203113275397) q[8];
h q[8];
rz(1.0660196368850254) q[8];
h q[8];
rz(3*pi) q[8];
cx q[5],q[8];
cx q[11],q[5];
rz(-1.5522082998690316) q[5];
h q[5];
rz(1.0660196368850254) q[5];
h q[5];
rz(3*pi) q[5];
cx q[11],q[5];
cx q[1],q[5];
rz(10.761926513095197) q[5];
cx q[1],q[5];
cx q[3],q[8];
rz(10.32362498466734) q[8];
cx q[3],q[8];
rz(1.863423564709623) q[9];
cx q[4],q[9];
cx q[10],q[4];
rz(-1.3743574391385005) q[4];
h q[4];
rz(1.0660196368850254) q[4];
h q[4];
rz(3*pi) q[4];
cx q[10],q[4];
cx q[4],q[8];
cx q[6],q[10];
rz(-1.4189497764777834) q[10];
h q[10];
rz(1.0660196368850254) q[10];
h q[10];
rz(3*pi) q[10];
cx q[6],q[10];
cx q[10],q[1];
rz(-1.9499946295220774) q[1];
h q[1];
rz(1.837205527712972) q[1];
h q[1];
cx q[10],q[1];
cx q[11],q[6];
rz(-1.3552031219537266) q[6];
h q[6];
rz(1.0660196368850254) q[6];
h q[6];
rz(3*pi) q[6];
cx q[11],q[6];
h q[11];
rz(5.217165670294561) q[11];
h q[11];
cx q[11],q[3];
rz(-2.105402619568344) q[3];
h q[3];
rz(1.837205527712972) q[3];
h q[3];
cx q[11],q[3];
cx q[2],q[6];
rz(10.248781961834098) q[6];
cx q[2],q[6];
rz(3.328987669378592) q[8];
cx q[4],q[8];
cx q[5],q[8];
rz(-2.6543390362715655) q[8];
h q[8];
rz(1.837205527712972) q[8];
h q[8];
cx q[5],q[8];
cx q[11],q[5];
rz(-2.5359561459786004) q[5];
h q[5];
rz(1.837205527712972) q[5];
h q[5];
cx q[11],q[5];
cx q[9],q[7];
rz(-1.6078383333243842) q[7];
h q[7];
rz(1.0660196368850254) q[7];
h q[7];
rz(3*pi) q[7];
cx q[9],q[7];
h q[9];
rz(5.217165670294561) q[9];
h q[9];
cx q[0],q[7];
rz(10.60154994994667) q[7];
cx q[0],q[7];
cx q[7],q[2];
rz(-1.663154065874302) q[2];
h q[2];
rz(1.837205527712972) q[2];
h q[2];
cx q[7],q[2];
cx q[2],q[3];
rz(7.118093784388968) q[3];
cx q[2],q[3];
cx q[3],q[8];
rz(7.221911953148744) q[8];
cx q[3],q[8];
cx q[9],q[0];
rz(-2.656766744676145) q[0];
h q[0];
rz(1.837205527712972) q[0];
h q[0];
cx q[9],q[0];
cx q[0],q[1];
rz(7.0549612548280445) q[1];
cx q[0],q[1];
cx q[1],q[5];
rz(7.3237437725907455) q[5];
cx q[1],q[5];
cx q[4],q[9];
rz(4.393320662181236) q[9];
cx q[4],q[9];
cx q[10],q[4];
rz(-2.116644146263753) q[4];
h q[4];
rz(1.837205527712972) q[4];
h q[4];
cx q[10],q[4];
cx q[4],q[8];
cx q[6],q[10];
rz(-2.2217777512679158) q[10];
h q[10];
rz(1.837205527712972) q[10];
h q[10];
cx q[6],q[10];
cx q[10],q[1];
rz(1.0067423042710635) q[1];
h q[1];
rz(1.34129923617233) q[1];
h q[1];
cx q[10],q[1];
cx q[11],q[6];
rz(-2.071484763796164) q[6];
h q[6];
rz(1.837205527712972) q[6];
h q[6];
cx q[11],q[6];
h q[11];
rz(1.837205527712972) q[11];
h q[11];
cx q[11],q[3];
rz(0.9706359314756039) q[3];
h q[3];
rz(1.34129923617233) q[3];
h q[3];
cx q[11],q[3];
cx q[2],q[6];
rz(7.204523464418198) q[6];
cx q[2],q[6];
rz(0.7734330119467349) q[8];
cx q[4],q[8];
cx q[5],q[8];
rz(0.843099999142976) q[8];
h q[8];
rz(1.34129923617233) q[8];
h q[8];
cx q[5],q[8];
cx q[11],q[5];
rz(0.8706042270031347) q[5];
h q[5];
rz(1.34129923617233) q[5];
h q[5];
cx q[11],q[5];
cx q[9],q[7];
rz(-2.667112894259456) q[7];
h q[7];
rz(1.837205527712972) q[7];
h q[7];
cx q[9],q[7];
h q[9];
rz(1.837205527712972) q[9];
h q[9];
cx q[0],q[7];
rz(7.286483037315993) q[7];
cx q[0],q[7];
cx q[7],q[2];
rz(1.0733847743322427) q[2];
h q[2];
rz(1.34129923617233) q[2];
h q[2];
cx q[7],q[2];
cx q[2],q[3];
rz(9.795219840546181) q[3];
cx q[2],q[3];
cx q[3],q[8];
rz(10.23192997564255) q[8];
cx q[3],q[8];
cx q[9],q[0];
rz(0.842535962862291) q[0];
h q[0];
rz(1.34129923617233) q[0];
h q[0];
cx q[9],q[0];
cx q[0],q[1];
rz(9.529653458063343) q[1];
cx q[0],q[1];
cx q[1],q[5];
rz(10.660284551309623) q[5];
cx q[1],q[5];
cx q[4],q[9];
rz(1.0207124716784957) q[9];
cx q[4],q[9];
cx q[10],q[4];
rz(0.9680241561509648) q[4];
h q[4];
rz(1.34129923617233) q[4];
h q[4];
cx q[10],q[4];
cx q[4],q[8];
cx q[6],q[10];
rz(0.94359817178246) q[10];
h q[10];
rz(1.34129923617233) q[10];
h q[10];
cx q[6],q[10];
cx q[10],q[1];
rz(1.0932592317556393) q[1];
h q[1];
rz(2.9983260191069885) q[1];
h q[1];
rz(3*pi) q[1];
cx q[10],q[1];
cx q[11],q[6];
rz(0.9785161617299707) q[6];
h q[6];
rz(1.34129923617233) q[6];
h q[6];
cx q[11],q[6];
h q[11];
rz(1.34129923617233) q[11];
h q[11];
cx q[11],q[3];
rz(0.9413781195215112) q[3];
h q[3];
rz(2.9983260191069885) q[3];
h q[3];
rz(3*pi) q[3];
cx q[11],q[3];
cx q[2],q[6];
rz(10.158785463912537) q[6];
cx q[2],q[6];
rz(3.253438576024256) q[8];
cx q[4],q[8];
cx q[5],q[8];
rz(0.40489943853554156) q[8];
h q[8];
rz(2.9983260191069885) q[8];
h q[8];
rz(3*pi) q[8];
cx q[5],q[8];
cx q[11],q[5];
rz(0.5205957102769965) q[5];
h q[5];
rz(2.9983260191069885) q[5];
h q[5];
rz(3*pi) q[5];
cx q[11],q[5];
cx q[9],q[7];
rz(0.8401322130052762) q[7];
h q[7];
rz(1.34129923617233) q[7];
h q[7];
cx q[9],q[7];
h q[9];
rz(1.34129923617233) q[9];
h q[9];
cx q[0],q[7];
rz(10.503547624334157) q[7];
cx q[0],q[7];
cx q[7],q[2];
rz(1.3735901454217778) q[2];
h q[2];
rz(2.9983260191069885) q[2];
h q[2];
rz(3*pi) q[2];
cx q[7],q[2];
cx q[9],q[0];
rz(0.4025268253091685) q[0];
h q[0];
rz(2.9983260191069885) q[0];
h q[0];
rz(3*pi) q[0];
cx q[9],q[0];
cx q[4],q[9];
rz(4.2936172610855445) q[9];
cx q[4],q[9];
cx q[10],q[4];
rz(0.9303917115691265) q[4];
h q[4];
rz(2.9983260191069885) q[4];
h q[4];
rz(3*pi) q[4];
cx q[10],q[4];
cx q[6],q[10];
rz(0.8276440416683588) q[10];
h q[10];
rz(2.9983260191069885) q[10];
h q[10];
rz(3*pi) q[10];
cx q[6],q[10];
cx q[11],q[6];
rz(0.9745262328054016) q[6];
h q[6];
rz(2.9983260191069885) q[6];
h q[6];
rz(3*pi) q[6];
cx q[11],q[6];
h q[11];
rz(3.2848592880725977) q[11];
h q[11];
cx q[9],q[7];
rz(0.3924154744990451) q[7];
h q[7];
rz(2.9983260191069885) q[7];
h q[7];
rz(3*pi) q[7];
cx q[9],q[7];
h q[9];
rz(3.2848592880725977) q[9];
h q[9];
