OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
cx q[1],q[4];
rz(6.572175969659431) q[4];
cx q[1],q[4];
h q[5];
cx q[0],q[5];
rz(6.212606286283044) q[5];
cx q[0],q[5];
cx q[2],q[5];
rz(5.49457697090435) q[5];
cx q[2],q[5];
h q[6];
cx q[3],q[6];
rz(5.580332888511968) q[6];
cx q[3],q[6];
cx q[4],q[6];
rz(6.1929532024813305) q[6];
cx q[4],q[6];
h q[7];
cx q[7],q[4];
rz(-1.1473549800327096) q[4];
h q[4];
rz(1.3396322554970839) q[4];
h q[4];
cx q[7],q[4];
h q[8];
cx q[0],q[8];
rz(5.755394056120286) q[8];
cx q[0],q[8];
cx q[2],q[8];
rz(4.958455979484866) q[8];
cx q[2],q[8];
h q[9];
cx q[3],q[9];
rz(5.396268347699618) q[9];
cx q[3],q[9];
h q[10];
h q[11];
cx q[11],q[5];
rz(-0.2667923099739653) q[5];
h q[5];
rz(1.3396322554970839) q[5];
h q[5];
cx q[11],q[5];
cx q[10],q[11];
rz(5.98859472595421) q[11];
cx q[10],q[11];
h q[12];
cx q[1],q[12];
rz(6.238059729568098) q[12];
cx q[1],q[12];
cx q[12],q[6];
rz(-0.6823672859658272) q[6];
h q[6];
rz(1.3396322554970839) q[6];
h q[6];
cx q[12],q[6];
cx q[12],q[11];
rz(-0.7849131189912333) q[11];
h q[11];
rz(1.3396322554970839) q[11];
h q[11];
cx q[12],q[11];
h q[12];
rz(1.3396322554970839) q[12];
h q[12];
h q[13];
cx q[7],q[13];
rz(5.198624719123917) q[13];
cx q[7],q[13];
cx q[9],q[13];
rz(6.5780150609222705) q[13];
cx q[9],q[13];
h q[14];
cx q[14],q[2];
rz(-0.005996263247094014) q[2];
h q[2];
rz(1.3396322554970839) q[2];
h q[2];
cx q[14],q[2];
cx q[10],q[14];
rz(5.050117527409122) q[14];
cx q[10],q[14];
h q[15];
cx q[15],q[1];
rz(-2.0699456329541945) q[1];
h q[1];
rz(1.3396322554970839) q[1];
h q[1];
cx q[15],q[1];
cx q[15],q[7];
rz(-0.23906075590833264) q[7];
h q[7];
rz(1.3396322554970839) q[7];
h q[7];
cx q[15],q[7];
cx q[15],q[10];
rz(-0.35980220123379514) q[10];
h q[10];
rz(1.3396322554970839) q[10];
h q[10];
cx q[15],q[10];
h q[15];
rz(1.3396322554970839) q[15];
h q[15];
h q[16];
cx q[16],q[9];
rz(-0.10519016125390834) q[9];
h q[9];
rz(1.3396322554970839) q[9];
h q[9];
cx q[16],q[9];
cx q[16],q[13];
rz(-1.4713332696177694) q[13];
h q[13];
rz(1.3396322554970839) q[13];
h q[13];
cx q[16],q[13];
cx q[16],q[14];
rz(0.40865192631238045) q[14];
h q[14];
rz(1.3396322554970839) q[14];
h q[14];
cx q[16],q[14];
h q[16];
rz(1.3396322554970839) q[16];
h q[16];
h q[17];
cx q[17],q[0];
rz(-0.15099440446281243) q[0];
h q[0];
rz(1.3396322554970839) q[0];
h q[0];
cx q[17],q[0];
cx q[17],q[3];
rz(-0.7454393594342736) q[3];
h q[3];
rz(1.3396322554970839) q[3];
h q[3];
cx q[17],q[3];
cx q[17],q[8];
rz(-1.0437617229823655) q[8];
h q[8];
rz(1.3396322554970839) q[8];
h q[8];
cx q[17],q[8];
h q[17];
rz(1.3396322554970839) q[17];
h q[17];
