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
cx q[0],q[5];
rz(1.7204311590851966) q[5];
cx q[0],q[5];
cx q[0],q[8];
rz(2.031947110159022) q[8];
cx q[0],q[8];
cx q[17],q[0];
rz(1.545324853634774) q[0];
cx q[17],q[0];
cx q[1],q[4];
rz(1.720847795966409) q[4];
cx q[1],q[4];
cx q[1],q[12];
rz(1.6795056907547834) q[12];
cx q[1],q[12];
cx q[15],q[1];
rz(1.8864294340501049) q[1];
cx q[15],q[1];
cx q[2],q[5];
rz(1.638651144722151) q[5];
cx q[2],q[5];
cx q[2],q[8];
rz(1.638617594992297) q[8];
cx q[2],q[8];
cx q[14],q[2];
rz(1.8107993219633551) q[2];
cx q[14],q[2];
cx q[3],q[6];
rz(1.5945102051814581) q[6];
cx q[3],q[6];
cx q[3],q[9];
rz(1.8855359886121439) q[9];
cx q[3],q[9];
cx q[17],q[3];
rz(1.700885747524264) q[3];
cx q[17],q[3];
cx q[4],q[6];
rz(1.657294351358785) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(1.6480277893753066) q[4];
cx q[7],q[4];
cx q[11],q[5];
rz(1.6020603966340152) q[5];
cx q[11],q[5];
cx q[12],q[6];
rz(1.4848208270983463) q[6];
cx q[12],q[6];
cx q[7],q[13];
rz(1.7063609405874265) q[13];
cx q[7],q[13];
cx q[15],q[7];
rz(1.9494536981389008) q[7];
cx q[15],q[7];
cx q[17],q[8];
rz(1.6124972170229865) q[8];
cx q[17],q[8];
cx q[9],q[13];
rz(1.531946119754471) q[13];
cx q[9],q[13];
cx q[16],q[9];
rz(1.4262701338944588) q[9];
cx q[16],q[9];
cx q[10],q[11];
rz(1.8267021418347924) q[11];
cx q[10],q[11];
cx q[10],q[14];
rz(1.842751129215373) q[14];
cx q[10],q[14];
cx q[15],q[10];
rz(2.005342124487972) q[10];
cx q[15],q[10];
cx q[12],q[11];
rz(1.811147231987031) q[11];
cx q[12],q[11];
cx q[16],q[13];
rz(1.8667953399324957) q[13];
cx q[16],q[13];
cx q[16],q[14];
rz(1.5852136929320269) q[14];
cx q[16],q[14];
rx(1.1008928590483038) q[0];
rx(1.1008928590483038) q[1];
rx(1.1008928590483038) q[2];
rx(1.1008928590483038) q[3];
rx(1.1008928590483038) q[4];
rx(1.1008928590483038) q[5];
rx(1.1008928590483038) q[6];
rx(1.1008928590483038) q[7];
rx(1.1008928590483038) q[8];
rx(1.1008928590483038) q[9];
rx(1.1008928590483038) q[10];
rx(1.1008928590483038) q[11];
rx(1.1008928590483038) q[12];
rx(1.1008928590483038) q[13];
rx(1.1008928590483038) q[14];
rx(1.1008928590483038) q[15];
rx(1.1008928590483038) q[16];
rx(1.1008928590483038) q[17];
cx q[0],q[5];
rz(1.809727231361633) q[5];
cx q[0],q[5];
cx q[0],q[8];
rz(2.137411891503215) q[8];
cx q[0],q[8];
cx q[17],q[0];
rz(1.6255323290063073) q[0];
cx q[17],q[0];
cx q[1],q[4];
rz(1.810165493076168) q[4];
cx q[1],q[4];
cx q[1],q[12];
rz(1.7666775957498493) q[12];
cx q[1],q[12];
cx q[15],q[1];
rz(1.9843413662990568) q[1];
cx q[15],q[1];
cx q[2],q[5];
rz(1.723702563537874) q[5];
cx q[2],q[5];
cx q[2],q[8];
rz(1.7236672724659814) q[8];
cx q[2],q[8];
cx q[14],q[2];
rz(1.904785801037671) q[2];
cx q[14],q[2];
cx q[3],q[6];
rz(1.6772705631158662) q[6];
cx q[3],q[6];
cx q[3],q[9];
rz(1.9834015480853053) q[9];
cx q[3],q[9];
cx q[17],q[3];
rz(1.7891673482397776) q[3];
cx q[17],q[3];
cx q[4],q[6];
rz(1.7433134143133031) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(1.7335658870879078) q[4];
cx q[7],q[4];
cx q[11],q[5];
rz(1.6852126344981073) q[5];
cx q[11],q[5];
cx q[12],q[6];
rz(1.5618879432069812) q[6];
cx q[12],q[6];
cx q[7],q[13];
rz(1.7949267219473755) q[13];
cx q[7],q[13];
cx q[15],q[7];
rz(2.050636798322427) q[7];
cx q[15],q[7];
cx q[17],q[8];
rz(1.6961911604141309) q[8];
cx q[17],q[8];
cx q[9],q[13];
rz(1.6114591945501757) q[13];
cx q[9],q[13];
cx q[16],q[9];
rz(1.5002982752062468) q[9];
cx q[16],q[9];
cx q[10],q[11];
rz(1.9215140299033249) q[11];
cx q[10],q[11];
cx q[10],q[14];
rz(1.9383960128557027) q[14];
cx q[10],q[14];
cx q[15],q[10];
rz(2.1094260190057144) q[10];
cx q[15],q[10];
cx q[12],q[11];
rz(1.9051517687432575) q[11];
cx q[12],q[11];
cx q[16],q[13];
rz(1.9636881977022684) q[13];
cx q[16],q[13];
cx q[16],q[14];
rz(1.6674915311065712) q[14];
cx q[16],q[14];
rx(0.5773835394828126) q[0];
rx(0.5773835394828126) q[1];
rx(0.5773835394828126) q[2];
rx(0.5773835394828126) q[3];
rx(0.5773835394828126) q[4];
rx(0.5773835394828126) q[5];
rx(0.5773835394828126) q[6];
rx(0.5773835394828126) q[7];
rx(0.5773835394828126) q[8];
rx(0.5773835394828126) q[9];
rx(0.5773835394828126) q[10];
rx(0.5773835394828126) q[11];
rx(0.5773835394828126) q[12];
rx(0.5773835394828126) q[13];
rx(0.5773835394828126) q[14];
rx(0.5773835394828126) q[15];
rx(0.5773835394828126) q[16];
rx(0.5773835394828126) q[17];
cx q[0],q[5];
rz(5.295709116155702) q[5];
cx q[0],q[5];
cx q[0],q[8];
rz(6.254595412313441) q[8];
cx q[0],q[8];
cx q[17],q[0];
rz(4.756709311848957) q[0];
cx q[17],q[0];
cx q[1],q[4];
rz(5.296991578239878) q[4];
cx q[1],q[4];
cx q[1],q[12];
rz(5.169735243515804) q[12];
cx q[1],q[12];
cx q[15],q[1];
rz(5.806673227306315) q[1];
cx q[15],q[1];
cx q[2],q[5];
rz(5.043979678860456) q[5];
cx q[2],q[5];
cx q[2],q[8];
rz(5.043876408462629) q[8];
cx q[2],q[8];
cx q[14],q[2];
rz(5.573873982815391) q[2];
cx q[14],q[2];
cx q[3],q[6];
rz(4.9081081709033345) q[6];
cx q[3],q[6];
cx q[3],q[9];
rz(5.803923086955966) q[9];
cx q[3],q[9];
cx q[17],q[3];
rz(5.235545817185179) q[3];
cx q[17],q[3];
cx q[4],q[6];
rz(5.1013658746513375) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(5.072842201087194) q[4];
cx q[7],q[4];
cx q[11],q[5];
rz(4.9313486344888044) q[5];
cx q[11],q[5];
cx q[12],q[6];
rz(4.570470110587653) q[6];
cx q[12],q[6];
cx q[7],q[13];
rz(5.25239916796541) q[13];
cx q[7],q[13];
cx q[15],q[7];
rz(6.000670044971201) q[7];
cx q[15],q[7];
cx q[17],q[8];
rz(4.963474514437958) q[8];
cx q[17],q[8];
cx q[9],q[13];
rz(4.715527842542034) q[13];
cx q[9],q[13];
cx q[16],q[9];
rz(4.390243521386644) q[9];
cx q[16],q[9];
cx q[10],q[11];
rz(5.622824914517031) q[11];
cx q[10],q[11];
cx q[10],q[14];
rz(5.672225768673614) q[14];
cx q[10],q[14];
cx q[15],q[10];
rz(6.1727018332478965) q[10];
cx q[15],q[10];
cx q[12],q[11];
rz(5.574944894763394) q[11];
cx q[12],q[11];
cx q[16],q[13];
rz(5.746236951982535) q[13];
cx q[16],q[13];
cx q[16],q[14];
rz(4.8794923065557345) q[14];
cx q[16],q[14];
rx(4.861604958697464) q[0];
rx(4.861604958697464) q[1];
rx(4.861604958697464) q[2];
rx(4.861604958697464) q[3];
rx(4.861604958697464) q[4];
rx(4.861604958697464) q[5];
rx(4.861604958697464) q[6];
rx(4.861604958697464) q[7];
rx(4.861604958697464) q[8];
rx(4.861604958697464) q[9];
rx(4.861604958697464) q[10];
rx(4.861604958697464) q[11];
rx(4.861604958697464) q[12];
rx(4.861604958697464) q[13];
rx(4.861604958697464) q[14];
rx(4.861604958697464) q[15];
rx(4.861604958697464) q[16];
rx(4.861604958697464) q[17];
cx q[0],q[5];
rz(6.043552754618198) q[5];
cx q[0],q[5];
cx q[0],q[8];
rz(7.137850003466442) q[8];
cx q[0],q[8];
cx q[17],q[0];
rz(5.428437067444393) q[0];
cx q[17],q[0];
cx q[1],q[4];
rz(6.045016322025607) q[4];
cx q[1],q[4];
cx q[1],q[12];
rz(5.899789241875367) q[12];
cx q[1],q[12];
cx q[15],q[1];
rz(6.626673634886074) q[1];
cx q[15],q[1];
cx q[2],q[5];
rz(5.7562748658944765) q[5];
cx q[2],q[5];
cx q[2],q[8];
rz(5.756157011970939) q[8];
cx q[2],q[8];
cx q[14],q[2];
rz(6.360999202160151) q[2];
cx q[14],q[2];
cx q[3],q[6];
rz(5.601216004431843) q[6];
cx q[3],q[6];
cx q[3],q[9];
rz(6.623535128233729) q[9];
cx q[3],q[9];
cx q[17],q[3];
rz(5.9748933809167655) q[3];
cx q[17],q[3];
cx q[4],q[6];
rz(5.821764962507035) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(5.789213264111344) q[4];
cx q[7],q[4];
cx q[11],q[5];
rz(5.627738414299879) q[5];
cx q[11],q[5];
cx q[12],q[6];
rz(5.215897743037973) q[6];
cx q[12],q[6];
cx q[7],q[13];
rz(5.994126709692638) q[13];
cx q[7],q[13];
cx q[15],q[7];
rz(6.84806608225616) q[7];
cx q[15],q[7];
cx q[17],q[8];
rz(5.664401011508803) q[8];
cx q[17],q[8];
cx q[9],q[13];
rz(5.381440078597364) q[13];
cx q[9],q[13];
cx q[16],q[9];
rz(5.01022011314355) q[9];
cx q[16],q[9];
cx q[10],q[11];
rz(6.416862832816158) q[11];
cx q[10],q[11];
cx q[10],q[14];
rz(6.473239922582607) q[14];
cx q[10],q[14];
cx q[15],q[10];
rz(7.044391666822302) q[10];
cx q[15],q[10];
cx q[12],q[11];
rz(6.362221344976411) q[11];
cx q[12],q[11];
cx q[16],q[13];
rz(6.557702735956293) q[13];
cx q[16],q[13];
cx q[16],q[14];
rz(5.568559096355808) q[14];
cx q[16],q[14];
rx(5.026505746687402) q[0];
rx(5.026505746687402) q[1];
rx(5.026505746687402) q[2];
rx(5.026505746687402) q[3];
rx(5.026505746687402) q[4];
rx(5.026505746687402) q[5];
rx(5.026505746687402) q[6];
rx(5.026505746687402) q[7];
rx(5.026505746687402) q[8];
rx(5.026505746687402) q[9];
rx(5.026505746687402) q[10];
rx(5.026505746687402) q[11];
rx(5.026505746687402) q[12];
rx(5.026505746687402) q[13];
rx(5.026505746687402) q[14];
rx(5.026505746687402) q[15];
rx(5.026505746687402) q[16];
rx(5.026505746687402) q[17];
