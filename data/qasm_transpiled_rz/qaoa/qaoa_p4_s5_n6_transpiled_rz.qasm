OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.48637249883504874) q[1];
cx q[0],q[1];
h q[2];
h q[3];
cx q[0],q[3];
rz(0.4902407999155768) q[3];
cx q[0],q[3];
cx q[2],q[3];
rz(0.45807517127879493) q[3];
cx q[2],q[3];
h q[4];
cx q[1],q[4];
rz(0.5127612141930012) q[4];
cx q[1],q[4];
cx q[2],q[4];
rz(0.505397312480701) q[4];
cx q[2],q[4];
cx q[4],q[3];
rz(0.447586679272737) q[3];
h q[3];
rz(0.9755432620508451) q[3];
h q[3];
cx q[4],q[3];
h q[5];
cx q[5],q[0];
rz(0.47107733916428973) q[0];
h q[0];
rz(0.9755432620508442) q[0];
h q[0];
cx q[5],q[0];
cx q[5],q[1];
rz(0.48725155347373317) q[1];
h q[1];
rz(0.9755432620508451) q[1];
h q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(8.80326116025018) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(8.823304259251039) q[3];
cx q[0],q[3];
cx q[1],q[4];
h q[4];
rz(0.9755432620508451) q[4];
h q[4];
rz(8.939990857193186) q[4];
cx q[1],q[4];
cx q[5],q[2];
rz(0.33471266303081926) q[2];
h q[2];
rz(0.9755432620508451) q[2];
h q[2];
cx q[5],q[2];
h q[5];
rz(0.9755432620508451) q[5];
h q[5];
cx q[2],q[3];
rz(2.373456930224937) q[3];
cx q[2],q[3];
cx q[2],q[4];
rz(2.618650450919793) q[4];
cx q[2],q[4];
cx q[4],q[3];
rz(-3.964073134322512) q[3];
h q[3];
rz(1.5775331445736755) q[3];
h q[3];
cx q[4],q[3];
cx q[5],q[0];
rz(-3.8423593349050122) q[0];
h q[0];
rz(1.5775331445736755) q[0];
h q[0];
cx q[5],q[0];
cx q[5],q[1];
rz(-3.7585547468141014) q[1];
h q[1];
rz(1.5775331445736755) q[1];
h q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(12.140324121106858) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(12.186908120936177) q[3];
cx q[0],q[3];
cx q[1],q[4];
h q[4];
rz(1.5775331445736755) q[4];
h q[4];
rz(6.174924809934339) q[4];
cx q[1],q[4];
cx q[5],q[2];
rz(-4.5489151709336815) q[2];
h q[2];
rz(1.5775331445736755) q[2];
h q[2];
cx q[5],q[2];
h q[5];
rz(1.5775331445736755) q[5];
h q[5];
cx q[2],q[3];
rz(5.5163683633834415) q[3];
cx q[2],q[3];
cx q[2],q[4];
rz(6.086245053894746) q[4];
cx q[2],q[4];
cx q[4],q[3];
rz(-0.8931245655786135) q[3];
h q[3];
rz(0.21841983376495033) q[3];
h q[3];
cx q[4],q[3];
cx q[5],q[0];
rz(-0.6102383905890125) q[0];
h q[0];
rz(0.21841983376495033) q[0];
h q[0];
cx q[5],q[0];
cx q[5],q[1];
rz(-0.4154604813866438) q[1];
h q[1];
rz(0.21841983376495033) q[1];
h q[1];
cx q[5],q[1];
cx q[0],q[1];
rz(10.34170197943866) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(10.373980869520324) q[3];
cx q[0],q[3];
cx q[1],q[4];
h q[4];
rz(0.21841983376495033) q[4];
h q[4];
rz(10.561901603176167) q[4];
cx q[1],q[4];
cx q[5],q[2];
rz(-2.252409441261123) q[2];
h q[2];
rz(0.21841983376495033) q[2];
h q[2];
cx q[5],q[2];
h q[5];
rz(0.21841983376495033) q[5];
h q[5];
cx q[2],q[3];
rz(3.8223907071962717) q[3];
cx q[2],q[3];
cx q[2],q[4];
rz(4.217268500441052) q[4];
cx q[2],q[4];
cx q[4],q[3];
rz(0.5932772341205856) q[3];
h q[3];
rz(1.0538783383739077) q[3];
h q[3];
rz(3*pi) q[3];
cx q[4],q[3];
h q[4];
rz(5.229306968805679) q[4];
h q[4];
cx q[5],q[0];
rz(0.7892941451107971) q[0];
h q[0];
rz(1.0538783383739077) q[0];
h q[0];
rz(3*pi) q[0];
cx q[5],q[0];
cx q[5],q[1];
rz(0.9242592564047074) q[1];
h q[1];
rz(1.0538783383739077) q[1];
h q[1];
rz(3*pi) q[1];
cx q[5],q[1];
cx q[5],q[2];
rz(-0.3485956675749229) q[2];
h q[2];
rz(1.0538783383739077) q[2];
h q[2];
rz(3*pi) q[2];
cx q[5],q[2];
h q[5];
rz(5.229306968805679) q[5];
h q[5];
