OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
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
cx q[0],q[6];
rz(1.6017336333676644) q[6];
cx q[0],q[6];
cx q[0],q[9];
rz(1.3142843614046502) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(1.3742745792713162) q[0];
cx q[12],q[0];
cx q[1],q[3];
rz(1.2876182147116393) q[3];
cx q[1],q[3];
cx q[1],q[9];
rz(1.2776301686284248) q[9];
cx q[1],q[9];
cx q[13],q[1];
rz(1.338635327539216) q[1];
cx q[13],q[1];
cx q[2],q[7];
rz(1.4658055030555168) q[7];
cx q[2],q[7];
cx q[2],q[8];
rz(1.3228191380338876) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(1.4018909741446879) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(1.390927494658417) q[5];
cx q[3],q[5];
cx q[13],q[3];
rz(1.493669435536741) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(1.3784586829404453) q[6];
cx q[4],q[6];
cx q[4],q[8];
rz(1.2431052070675435) q[8];
cx q[4],q[8];
cx q[9],q[4];
rz(1.5196265696091809) q[4];
cx q[9],q[4];
cx q[5],q[12];
rz(1.4272520608424357) q[12];
cx q[5],q[12];
cx q[13],q[5];
rz(1.321042504329678) q[5];
cx q[13],q[5];
cx q[8],q[6];
rz(1.4017589947633224) q[6];
cx q[8],q[6];
cx q[7],q[10];
rz(1.3931724156054668) q[10];
cx q[7],q[10];
cx q[11],q[7];
rz(1.262751514845488) q[7];
cx q[11],q[7];
cx q[11],q[10];
rz(1.4794217130627423) q[10];
cx q[11],q[10];
cx q[12],q[11];
rz(1.3338172351252502) q[11];
cx q[12],q[11];
rx(2.687526717400007) q[0];
rx(2.687526717400007) q[1];
rx(2.687526717400007) q[2];
rx(2.687526717400007) q[3];
rx(2.687526717400007) q[4];
rx(2.687526717400007) q[5];
rx(2.687526717400007) q[6];
rx(2.687526717400007) q[7];
rx(2.687526717400007) q[8];
rx(2.687526717400007) q[9];
rx(2.687526717400007) q[10];
rx(2.687526717400007) q[11];
rx(2.687526717400007) q[12];
rx(2.687526717400007) q[13];
cx q[0],q[6];
rz(1.5200327042254977) q[6];
cx q[0],q[6];
cx q[0],q[9];
rz(1.2472455908832276) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(1.3041758389540155) q[0];
cx q[12],q[0];
cx q[1],q[3];
rz(1.221939626005766) q[3];
cx q[1],q[3];
cx q[1],q[9];
rz(1.212461048306254) q[9];
cx q[1],q[9];
cx q[13],q[1];
rz(1.2703544674985017) q[1];
cx q[13],q[1];
cx q[2],q[7];
rz(1.3910379705229416) q[7];
cx q[2],q[7];
cx q[2],q[8];
rz(1.2553450272248523) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(1.3303835819306495) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(1.3199793255523908) q[5];
cx q[3],q[5];
cx q[13],q[3];
rz(1.4174806247554945) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(1.3081465206469347) q[6];
cx q[4],q[6];
cx q[4],q[8];
rz(1.1796971295176286) q[8];
cx q[4],q[8];
cx q[9],q[4];
rz(1.4421137421953265) q[4];
cx q[9],q[4];
cx q[5],q[12];
rz(1.3544510550686295) q[12];
cx q[5],q[12];
cx q[13],q[5];
rz(1.2536590157198368) q[5];
cx q[13],q[5];
cx q[8],q[6];
rz(1.330258334528847) q[6];
cx q[8],q[6];
cx q[7],q[10];
rz(1.3221097379922808) q[10];
cx q[7],q[10];
cx q[11],q[7];
rz(1.1983413221084829) q[7];
cx q[11],q[7];
cx q[11],q[10];
rz(1.4039596474406382) q[10];
cx q[11],q[10];
cx q[12],q[11];
rz(1.2657821354398866) q[11];
cx q[12],q[11];
rx(1.3891683687902008) q[0];
rx(1.3891683687902008) q[1];
rx(1.3891683687902008) q[2];
rx(1.3891683687902008) q[3];
rx(1.3891683687902008) q[4];
rx(1.3891683687902008) q[5];
rx(1.3891683687902008) q[6];
rx(1.3891683687902008) q[7];
rx(1.3891683687902008) q[8];
rx(1.3891683687902008) q[9];
rx(1.3891683687902008) q[10];
rx(1.3891683687902008) q[11];
rx(1.3891683687902008) q[12];
rx(1.3891683687902008) q[13];
cx q[0],q[6];
rz(6.518540830843034) q[6];
cx q[0],q[6];
cx q[0],q[9];
rz(5.348714726768894) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(5.5928556229008235) q[0];
cx q[12],q[0];
cx q[1],q[3];
rz(5.240192084552677) q[3];
cx q[1],q[3];
cx q[1],q[9];
rz(5.199543948771893) q[9];
cx q[1],q[9];
cx q[13],q[1];
rz(5.4478153285867545) q[1];
cx q[13],q[1];
cx q[2],q[7];
rz(5.965357049818877) q[7];
cx q[2],q[7];
cx q[2],q[8];
rz(5.3834485231884885) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(5.70524546964723) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(5.660627633578502) q[5];
cx q[3],q[5];
cx q[13],q[3];
rz(6.07875429503051) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(5.60988357938467) q[6];
cx q[4],q[6];
cx q[4],q[8];
rz(5.0590384571411064) q[8];
cx q[4],q[8];
cx q[9],q[4];
rz(6.184391483872649) q[4];
cx q[9],q[4];
cx q[5],q[12];
rz(5.808456937340668) q[12];
cx q[5],q[12];
cx q[13],q[5];
rz(5.376218195310567) q[5];
cx q[13],q[5];
cx q[8],q[6];
rz(5.704708356004651) q[6];
cx q[8],q[6];
cx q[7],q[10];
rz(5.669763739951315) q[10];
cx q[7],q[10];
cx q[11],q[7];
rz(5.138992612287728) q[7];
cx q[11],q[7];
cx q[11],q[10];
rz(6.020770646090074) q[10];
cx q[11],q[10];
cx q[12],q[11];
rz(5.428207241778227) q[11];
cx q[12],q[11];
rx(1.619080693323163) q[0];
rx(1.619080693323163) q[1];
rx(1.619080693323163) q[2];
rx(1.619080693323163) q[3];
rx(1.619080693323163) q[4];
rx(1.619080693323163) q[5];
rx(1.619080693323163) q[6];
rx(1.619080693323163) q[7];
rx(1.619080693323163) q[8];
rx(1.619080693323163) q[9];
rx(1.619080693323163) q[10];
rx(1.619080693323163) q[11];
rx(1.619080693323163) q[12];
rx(1.619080693323163) q[13];
cx q[0],q[6];
rz(4.416547334365172) q[6];
cx q[0],q[6];
cx q[0],q[9];
rz(3.62394781007083) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(3.789362103238884) q[0];
cx q[12],q[0];
cx q[1],q[3];
rz(3.550419792277266) q[3];
cx q[1],q[3];
cx q[1],q[9];
rz(3.5228792091332446) q[9];
cx q[1],q[9];
cx q[13],q[1];
rz(3.6910920544884953) q[1];
cx q[13],q[1];
cx q[2],q[7];
rz(4.041745301686899) q[7];
cx q[2],q[7];
cx q[2],q[8];
rz(3.6474812142436623) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(3.865510292064964) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(3.835280093302982) q[5];
cx q[3],q[5];
cx q[13],q[3];
rz(4.1185760394333135) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(3.8008991600388624) q[6];
cx q[4],q[6];
cx q[4],q[8];
rz(3.4276816533260552) q[8];
cx q[4],q[8];
cx q[9],q[4];
rz(4.190149058134531) q[4];
cx q[9],q[4];
cx q[5],q[12];
rz(3.9354397968953294) q[12];
cx q[5],q[12];
cx q[13],q[5];
rz(3.6425824054236386) q[5];
cx q[13],q[5];
cx q[8],q[6];
rz(3.8651463781326987) q[6];
cx q[8],q[6];
cx q[7],q[10];
rz(3.841470136027943) q[10];
cx q[7],q[10];
cx q[11],q[7];
rz(3.481853487168594) q[7];
cx q[11],q[7];
cx q[11],q[10];
rz(4.079290018710247) q[10];
cx q[11],q[10];
cx q[12],q[11];
rz(3.677806866012503) q[11];
cx q[12],q[11];
rx(0.40752659188707335) q[0];
rx(0.40752659188707335) q[1];
rx(0.40752659188707335) q[2];
rx(0.40752659188707335) q[3];
rx(0.40752659188707335) q[4];
rx(0.40752659188707335) q[5];
rx(0.40752659188707335) q[6];
rx(0.40752659188707335) q[7];
rx(0.40752659188707335) q[8];
rx(0.40752659188707335) q[9];
rx(0.40752659188707335) q[10];
rx(0.40752659188707335) q[11];
rx(0.40752659188707335) q[12];
rx(0.40752659188707335) q[13];
cx q[0],q[6];
rz(5.638966263332551) q[6];
cx q[0],q[6];
cx q[0],q[9];
rz(4.626989794054733) q[9];
cx q[0],q[9];
cx q[12],q[0];
rz(4.838187715877012) q[0];
cx q[12],q[0];
cx q[1],q[3];
rz(4.533110575661341) q[3];
cx q[1],q[3];
cx q[1],q[9];
rz(4.4979472665276985) q[9];
cx q[1],q[9];
cx q[13],q[1];
rz(4.712718328220345) q[1];
cx q[13],q[1];
cx q[2],q[7];
rz(5.160425933592139) q[7];
cx q[2],q[7];
cx q[2],q[8];
rz(4.65703681090869) q[8];
cx q[2],q[8];
cx q[10],q[2];
rz(4.935412320369079) q[2];
cx q[10],q[2];
cx q[3],q[5];
rz(4.896814959569558) q[5];
cx q[3],q[5];
cx q[13],q[3];
rz(5.258522003969986) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(4.852918017433394) q[6];
cx q[4],q[6];
cx q[4],q[8];
rz(4.376400781251409) q[8];
cx q[4],q[8];
cx q[9],q[4];
rz(5.349905115542376) q[4];
cx q[9],q[4];
cx q[5],q[12];
rz(5.0246970237122754) q[12];
cx q[5],q[12];
cx q[13],q[5];
rz(4.650782102065952) q[5];
cx q[13],q[5];
cx q[8],q[6];
rz(4.934947681765337) q[6];
cx q[8],q[6];
cx q[7],q[10];
rz(4.9047182920716335) q[10];
cx q[7],q[10];
cx q[11],q[7];
rz(4.445566380606372) q[7];
cx q[11],q[7];
cx q[11],q[10];
rz(5.208362336540588) q[10];
cx q[11],q[10];
cx q[12],q[11];
rz(4.695756044348683) q[11];
cx q[12],q[11];
rx(1.3531230919525346) q[0];
rx(1.3531230919525346) q[1];
rx(1.3531230919525346) q[2];
rx(1.3531230919525346) q[3];
rx(1.3531230919525346) q[4];
rx(1.3531230919525346) q[5];
rx(1.3531230919525346) q[6];
rx(1.3531230919525346) q[7];
rx(1.3531230919525346) q[8];
rx(1.3531230919525346) q[9];
rx(1.3531230919525346) q[10];
rx(1.3531230919525346) q[11];
rx(1.3531230919525346) q[12];
rx(1.3531230919525346) q[13];
