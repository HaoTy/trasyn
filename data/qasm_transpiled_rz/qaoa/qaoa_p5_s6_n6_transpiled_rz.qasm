OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
h q[0];
h q[1];
cx q[0],q[1];
rz(4.989089171655285) q[1];
cx q[0],q[1];
h q[2];
cx q[0],q[2];
rz(4.366161022083563) q[2];
cx q[0],q[2];
h q[3];
cx q[2],q[3];
rz(4.151992442512679) q[3];
cx q[2],q[3];
h q[4];
cx q[4],q[0];
rz(-1.781391523530317) q[0];
h q[0];
rz(1.162267050536613) q[0];
h q[0];
cx q[4],q[0];
cx q[1],q[4];
rz(4.782501174801589) q[4];
cx q[1],q[4];
cx q[3],q[4];
rz(-1.8341509415223038) q[4];
h q[4];
rz(1.162267050536613) q[4];
h q[4];
cx q[3],q[4];
h q[5];
cx q[5],q[1];
rz(-2.615552724679396) q[1];
h q[1];
rz(1.162267050536613) q[1];
h q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(11.077400353009349) q[1];
cx q[0],q[1];
cx q[5],q[2];
rz(-1.2839002097784595) q[2];
h q[2];
rz(1.162267050536613) q[2];
h q[2];
cx q[5],q[2];
cx q[0],q[2];
rz(10.478803814764511) q[2];
cx q[0],q[2];
cx q[4],q[0];
rz(-1.9572318619270677) q[0];
h q[0];
rz(1.1968662610269725) q[0];
h q[0];
cx q[4],q[0];
cx q[1],q[4];
rz(4.595696388670699) q[4];
cx q[1],q[4];
cx q[5],q[3];
rz(-2.2326830789618732) q[3];
h q[3];
rz(1.162267050536613) q[3];
h q[3];
cx q[5],q[3];
h q[5];
rz(1.162267050536613) q[5];
h q[5];
cx q[2],q[3];
rz(10.273000672909252) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-2.0079304938503943) q[4];
h q[4];
rz(1.1968662610269725) q[4];
h q[4];
cx q[3],q[4];
cx q[5],q[1];
rz(-2.7588106759352167) q[1];
h q[1];
rz(1.1968662610269725) q[1];
h q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(10.746821030753562) q[1];
cx q[0],q[1];
cx q[5],q[2];
rz(-1.4791725890823928) q[2];
h q[2];
rz(1.1968662610269725) q[2];
h q[2];
cx q[5],q[2];
cx q[0],q[2];
rz(10.189499995594424) q[2];
cx q[0],q[2];
cx q[4],q[0];
rz(-2.255522769692167) q[0];
h q[0];
rz(0.5034825591702052) q[0];
h q[0];
cx q[4],q[0];
cx q[1],q[4];
rz(4.278805681237451) q[4];
cx q[1],q[4];
cx q[5],q[3];
rz(-2.3908959418152307) q[3];
h q[3];
rz(1.1968662610269725) q[3];
h q[3];
cx q[5],q[3];
h q[5];
rz(1.1968662610269725) q[5];
h q[5];
cx q[2],q[3];
rz(9.997887761427336) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-2.3027255385346157) q[4];
h q[4];
rz(0.5034825591702052) q[4];
h q[4];
cx q[3],q[4];
cx q[5],q[1];
rz(-3.0018296823491433) q[1];
h q[1];
rz(0.5034825591702052) q[1];
h q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(0.3355244720771271) q[1];
cx q[0],q[1];
cx q[5],q[2];
rz(-1.8104274980248176) q[2];
h q[2];
rz(0.5034825591702052) q[2];
h q[2];
cx q[5],q[2];
cx q[0],q[2];
rz(12.075495618820305) q[2];
cx q[0],q[2];
cx q[4],q[0];
rz(-0.3109395695571555) q[0];
h q[0];
rz(2.742268716466465) q[0];
h q[0];
cx q[4],q[0];
cx q[1],q[4];
rz(6.344642520082239) q[4];
cx q[1],q[4];
cx q[5],q[3];
rz(-2.6592840654445187) q[3];
h q[3];
rz(0.5034825591702052) q[3];
h q[3];
cx q[5],q[3];
h q[5];
rz(0.5034825591702052) q[5];
h q[5];
cx q[2],q[3];
rz(11.79137167890984) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.38093216014771425) q[4];
h q[4];
rz(2.742268716466465) q[4];
h q[4];
cx q[3],q[4];
cx q[5],q[1];
rz(-1.4175685973342134) q[1];
h q[1];
rz(2.742268716466465) q[1];
h q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(8.00871662296288) q[1];
cx q[0],q[1];
cx q[5],q[2];
rz(0.3490507633741382) q[2];
h q[2];
rz(2.742268716466465) q[2];
h q[2];
cx q[5],q[2];
cx q[0],q[2];
rz(7.793270076992982) q[2];
cx q[0],q[2];
cx q[4],q[0];
rz(1.556994850887401) q[0];
h q[0];
rz(0.4618411302811243) q[0];
h q[0];
cx q[4],q[0];
cx q[1],q[4];
rz(1.6540805868483905) q[4];
cx q[1],q[4];
cx q[5],q[3];
rz(-0.9096395984523697) q[3];
h q[3];
rz(2.742268716466465) q[3];
h q[3];
cx q[5],q[3];
h q[5];
rz(2.742268716466465) q[5];
h q[5];
cx q[2],q[3];
rz(7.719197520221306) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(1.5387474263945915) q[4];
h q[4];
rz(0.4618411302811243) q[4];
h q[4];
cx q[3],q[4];
cx q[5],q[1];
rz(1.2684910327612986) q[1];
h q[1];
rz(0.4618411302811243) q[1];
h q[1];
cx q[5],q[1];
cx q[5],q[2];
rz(-4.554127618432708) q[2];
h q[2];
rz(0.4618411302811243) q[2];
h q[2];
cx q[5],q[2];
cx q[5],q[3];
rz(1.400910707138305) q[3];
h q[3];
rz(0.4618411302811243) q[3];
h q[3];
cx q[5],q[3];
h q[5];
rz(0.4618411302811243) q[5];
h q[5];
